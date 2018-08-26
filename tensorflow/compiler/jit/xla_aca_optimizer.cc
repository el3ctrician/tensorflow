#include "tensorflow/compiler/jit/xla_aca_optimizer.h"

#include <atomic>
#include <deque>
#include <unordered_map>
#include <unordered_set>

#include "tensorflow/compiler/jit/deadness_analysis.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/graphcycles/graphcycles.h"
#include "tensorflow/compiler/jit/union_find.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"

namespace tensorflow {

// Is 'node' an operator that consumes only the shape of its input, not the
// data itself?
static bool IsShapeConsumerOp(const Node& node) {
  return node.type_string() == "Shape" || node.type_string() == "ShapeN" ||
         node.type_string() == "Rank" || node.type_string() == "Size";
}

// Returns true if the op can be decomposed into XLA ops for which
// there are fusable elemental implementations.
bool IsXlaFusable(const NodeDef& node) {
  static const std::unordered_set<std::string>* elementwise_ops =
      new std::unordered_set<std::string>(
          {
           // tf2xla/kernels/binary_ops.cc
           "Add", "Sub", "Mul", "Div"
           });

  return elementwise_ops->count(node.op()) > 0;
}

Status XlaACAOptimizer::Optimize(grappler::Cluster* cluster,
                                    const grappler::GrapplerItem& item,
                                    GraphDef* output) {
  VLOG(2) << "ACA optimizer!";

  // Create a Graph out of GraphDef. This is required currently because the
  // helpers around clustering, encapsulation etc work on graphs.
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             item.graph.library());
  Graph graph(function_library);
  ShapeRefiner shape_refiner(graph.versions(), graph.op_registry());
  shape_refiner.set_require_shape_inference_fns(false);
  shape_refiner.set_disable_constant_propagation(true);
  ImportGraphDefOptions options;
  // Graph optimization happens at the late stage of graph execution, when
  // colocation constraints are already validated previously and the device
  // placement of nodes has also completed, so there is no need to validate
  // colocation constraints again.
  options.validate_colocation_constraints = false;
  options.validate_shape = false;
  TF_RETURN_IF_ERROR(
      ImportGraphDef(options, item.graph, &graph, &shape_refiner));

  std::unique_ptr<DeadnessAnalysis> deadness;
  TF_RETURN_IF_ERROR(DeadnessAnalysis::Run(graph, &deadness));

  // Collect nodes that can be fused via XLA, while ignoring those that
  // explicitly ask for XLA: (*) nodes that are marked to be compiled
  // explicitly. (*) nodes assigned to XLA device.
  OrderedNodeSet compilation_candidates;
  for (Node* node : graph.op_nodes()) {
    // If there is a _XlaCompile annotation, ignore the node if it is
    // true. Nodes are marked with this attr via experimental_jit_scope, and
    // will be handled by the mark_for_compilation pass.
    bool compile = false;
    Status status = GetNodeAttr(node->attrs(), kXlaCompileAttr, &compile);
    if (status.ok() && compile) {
      continue;
    }
    // If there is already a _XlaCluster annotation, ignore the node. Nodes are
    // marked with this attr to indicate they are already part of a cluster and
    // hence ignored.
    status = GetNodeAttr(node->attrs(), kXlaClusterAttr, &compile);
    if (status.ok()) {
      continue;
    }

    // If there is an explicit XLA device placement, ignore the node.
    DeviceType device_type("");
    TF_RETURN_IF_ERROR(DeviceToDeviceType(node->def().device(), &device_type));
    if (device_type.type_string().find("XLA") != string::npos) continue;

    // Assume all fusable ops are registered.
    // TODO(hpucha): Check for registration if possible.
    if (!IsXlaFusable(node->def())) {
      continue;
    }

    // XLA does not offer guaranteed aliasing between the input and output of
    // the XLA cluster so it can't implement the forward-tensor-ref semantic.
    // Leave such nodes out of XLA clusters.
    if (HasForwardedRefInput(*node)) {
      continue;
    }

    // If inputs to `node` can have conflicting deadness (i.e. some are alive
    // and some are dead) then don't compile it.  XLA cannot represent the
    // deadness semantics of these nodes correctly and auto-clustering these
    // nodes can cause deadness to propagate to nodes that should be live.
    if (node->IsMerge() || deadness->HasInputsWithMismatchingDeadness(*node)) {
      continue;
    }

    compilation_candidates.insert(node);
  }

  if (compilation_candidates.empty()) {
    VLOG(2) << "No compilable candidates";
    *output = item.graph;
    return Status::OK();
  }

  GraphCycles cycles;
  TF_RETURN_IF_ERROR(CreateCycleDetectionGraph(&graph, &cycles));
  TF_RETURN_IF_ERROR(AdjustCycleDetectionGraphForResourceOps(
      &graph, &graph.flib_def(), /*resource_ops_to_ignore=*/{}, &cycles));


  struct Cluster {
    // Identifies the node that represents this cluster in the cycle detection
    // graph.
    int representative = -1;
  };

  // Each compilation candidate belongs to a cluster. The cluster's
  // representative names the node in the 'cycles' graph that represents the
  // cluster.
  std::vector<UnionFind<Cluster>> clusters(graph.num_node_ids());
  std::deque<UnionFind<Cluster>*> worklist;
  for (Node* node : compilation_candidates) {
    Cluster& cluster = clusters[node->id()].Get();
    cluster.representative = node->id();
    worklist.push_back(&clusters[node->id()]);
  }

  // Repeatedly contract edges between clusters that are on the same device,
  // provided the contraction would not create a cycle. This is a simplified
  // version of the clustering in mark_for_compilation_pass that also deals with
  // nodes that are explicitly tagged to be compiled/clustered.
  while (!worklist.empty()) {
    int from = worklist.front()->Get().representative;
    worklist.pop_front();

    Node* node_from = graph.FindNodeId(from);
    if (node_from->IsControlFlow()) {
      // Control flow nodes aren't compilation candidates and should never
      // appear.
      return errors::Internal(
          "Found control flow node in clustering worklist: ",
          node_from->type_string());
    }
    for (int to : cycles.Successors(from)) {
      if (to >= graph.num_node_ids()) {
        // Node is a "frame" node that is present only in the cycle detection
        // graph. No clustering is possible.
        continue;
      }
      Node* node_to = graph.FindNodeId(to);
      if (compilation_candidates.find(node_to) ==
          compilation_candidates.cend()) {
        continue;
      }

      // Do not cluster across devices.
      if (node_from->def().device() != node_to->def().device()) {
        VLOG(2) << "Devices " << node_from->def().device() << " "
                << node_to->def().device();
        VLOG(2) << "Device names " << node_from->assigned_device_name() << " "
                << node_to->assigned_device_name();
        continue;
      }

      // Ops that consume shapes cannot be the root of a cluster. This is an
      // optimization.
      if (clusters[from].Size() == 1 && IsShapeConsumerOp(*node_from)) {
        continue;
      }

      // If contracting the edge would create a cycle, bail out.
      // However, just because we can't merge the clusters now does not mean
      // we won't be able to merge them in the future.
      // e.g., if we have edges 1->2, 2->3 and 1->3, we cannot contract edge
      // 1->3. But if we first contract 1->2 then we can later contract 1->3.
      if (!cycles.ContractEdge(from, to)) continue;

      // Merge the clusters. ContractEdge uses 'from' as the number of the
      // merged node, so make sure 'from' is the chosen representative.
      clusters[from].Merge(&clusters[to]);

      worklist.push_back(&clusters[from]);
      break;
    }
  }

  // Count the number of non-trivial elements in each cluster.
  std::vector<int> effective_cluster_sizes(graph.num_node_ids());
  for (const Node* n : compilation_candidates) {
    int cluster = clusters[n->id()].Get().representative;
    // Identity nodes will be removed if the node gets marked for compilation.
    // Therefore we don't want to count them towards the effective cluster size.
    if (n->def().op() != "Identity") {
      effective_cluster_sizes[cluster]++;
    }
  }

  const int min_cluster_size = 2;
  int num_clusters = 0;
  for (auto size : effective_cluster_sizes) {
    if (size >= min_cluster_size) {
      VLOG(3) << "Cluster " << num_clusters << " " << size;
      num_clusters++;
    }
  }

  // Names for each cluster.
  std::unordered_map<int, string> cluster_names;
  // Sequence number generator to ensure clusters have unique names.
  static std::atomic<int64> cluster_sequence_num;

  for (Node* n : compilation_candidates) {
    int cluster = clusters[n->id()].Get().representative;

    // Compile if this is a cluster of >= min_cluster_size compilable operators.
    if (effective_cluster_sizes[cluster] >= min_cluster_size) {
      string& name = cluster_names[cluster];

      if (name.empty()) {
        name = strings::StrCat("cluster_", cluster_sequence_num++);
      }
      n->AddAttr(kXlaClusterAttr, name);
      VLOG(3) << "Assigning node " << n->name() << " to cluster " << name;
    }
  }

  graph.ToGraphDef(output);
  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(XlaACAOptimizer, "xla-aca-optimizer");

} // namespace tensorflow