/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/jit/xla_aca_pass.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

  Status XlaACAPass::Run(const GraphOptimizationPassOptions& options) {
    VLOG(1) << "ACA_Project : -------------------------------START------------------------------------";
    VLOG(1) << "ACA_Project : Start of our new compiler pass";
    Graph* graph = options.graph->get();
    VLOG(1) << "ACA_Project : loaded graph";
    VLOG(1) << "ACA_Project : looping through all nodes";
    // Loop through our graph nodes !.
    for (Node* n : graph->op_nodes()) {
      VLOG(1) << "ACA_Project : -------------------------Node Analysis---------------------------";
      VLOG(1) << "ACA_Project : node op is : " << n->type_string();
      //VLOG(1) << "ACA_Project : node summary :" << SummarizeNode(*n);
      VLOG(1) << "ACA_Project : node num_inputs :" << n->num_inputs();
      //make a forloop to get all input nodes type, check if we have a bias node with a 2DConv as input to make the optimization
      VLOG(1) << "ACA_Project : ------------------------End Node Analysis--------------------------";
    }
    VLOG(1) << "ACA_Project : -----------------------------END---------------------------------";

  //   for (Node* n : graph->op_nodes()) {
  //     // In all cases, only try to compile computational nodes.
  //     if (n->IsSend() || n->IsRecv() || n->IsControlFlow()) {
  //       continue;
  //     }

  //     // Only compile nodes that are marked for compilation by the
  //     // compilation-marking pass (via 'attr_name').
  //     if (IsXlaCompiledKernel(*n)) {
  //       TF_RETURN_IF_ERROR(ReplaceNodeWithXlaLaunch(graph, n));
  //     }
  //   }

  //   if (VLOG_IS_ON(1)) {
  //     dump_graph::DumpGraphToFile("build_xla_launch_ops", *graph,
  //                                 options.flib_def);
  //   }
     return Status::OK();
  }
}  // namespace tensorflow
