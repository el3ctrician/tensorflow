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
    Graph* graph_out = options.graph->get();
    VLOG(1) << "ACA_Project : loaded graph";
    VLOG(1) << "ACA_Project : looping through all nodes";


    // Loop through our graph nodes !.
    for (Node* n : graph_out->op_nodes()) {
      VLOG(1) << "ACA_Project : -------------------------Node Analysis---------------------------";
      VLOG(1) << "ACA_Project : node op is : " << n->type_string();
      //VLOG(1) << "ACA_Project : node summary :" << SummarizeNode(*n);
      VLOG(1) << "ACA_Project : node num_inputs :" << n->num_inputs();
      
      NodeDef node_def;
      Status status;
      Node* new_node = graph_out->AddNode(node_def, &status);
      
      //Find an Add Operation
      if(n->name() == "Add"){
        VLOG(1) << "ACA_Project : -------------------------Node Input Edges Analysis---------------------------";

        const Edge* edges[10];
        int i=0;

        // Loop through the input edges
        for (const Edge* edge : n->in_edges()) {
          VLOG(1) << "    ACA_Project : input node/edge op is : " << edge->src()->type_string();
          edges[i++] = edge; //store all the edges of the Add operation

          if(edge->src()->type_string() == "MatMul"){
              VLOG(1) << "      ACA_Project : -------------------------Node Input Edge of an Edge Analysis---------------------------";

              const Edge* subedges[10]; //store all subedges of the edges of the Add operation
              int j=0;
              // Loop through the input edges of the edges
              for (const Edge* subedge : edge->src()->in_edges()){
                VLOG(1) << "          ACA_Project : input node/edge op is : " << subedge->src()->type_string();
                subedges[j++] = subedge;
              }

              //Create a new operation that has 2 of the "MatMul" node inputs, the other 
              //input of "Add" node and it's output
              //It would be a problem if more than one edge of Add where to be a MatMul operation

              //### bisogna settare l'operazione da qualche parte, forse con node_def!!
              graph_out->AddEdge(subedges[0]->src(), subedges[0]->dst_input(), new_node, 0);
              graph_out->AddEdge(subedges[1]->src(), subedges[1]->dst_input(), new_node, 1);

              VLOG(1) << "      ACA_Project : -------------------------END Node Input Edge of an Edge Analysis---------------------------";
          }

          //Bisogna aggiungere il secondo input del nodo principale
          graph_out->AddEdge(edge->src(), edges[1]->dst_input(), new_node, 3);
              
        }

        //remove node and edge after setted up the new node
        graph_out->RemoveNode(n);
        graph_out->RemoveEdge(edges[0]);

        VLOG(1) << "ACA_Project : -------------------------END Node Input Edges Analysis---------------------------";
      }
      VLOG(1) << "ACA_Project : ------------------------End Node Analysis--------------------------";
    }
    VLOG(1) << "ACA_Project : -----------------------------END---------------------------------";

    //Print again everything so that we can verify
    VLOG(1) << "ACA_Project : ################# NEW GRAPH #################";
    // Loop through our graph nodes !.
    for (Node* n : graph_out->op_nodes()) {
      VLOG(1) << "ACA_Project : +++++++++++++Node Analysis+++++++++++++";
      VLOG(1) << "ACA_Project : node op is : " << n->type_string();
      VLOG(1) << "ACA_Project : node num_inputs :" << n->num_inputs();
 
      //Find an Add Operation
      if(n->name() == "Add"){
        VLOG(1) << "ACA_Project : +++++++++++++Node Input Edges Analysis+++++++++++++";

        // Loop through the input edges
        for (const Edge* edge : n->in_edges()) {
          VLOG(1) << "    ACA_Project : input node/edge op is : " << edge->src()->type_string();

          if(edge->src()->type_string() == "MatMul"){
              VLOG(1) << "      ACA_Project : +++++++++++++Node Input Edge of an Edge Analysis+++++++++++++";
              // Loop through the input edges of the edges
              for (const Edge* subedge : edge->src()->in_edges()){
                VLOG(1) << "          ACA_Project : input node/edge op is : " << subedge->src()->type_string();
              }
              VLOG(1) << "      ACA_Project : +++++++++++++END Node Input Edge of an Edge Analysis+++++++++++++";
          } 
        }
        VLOG(1) << "ACA_Project : +++++++++++++END Node Input Edges Analysis+++++++++++++";
      }
      VLOG(1) << "ACA_Project : +++++++++++++End Node Analysis+++++++++++++";
    }
    VLOG(1) << "ACA_Project : ++++++++++++++++END++++++++++++++++++++++++++";


      //update the graph
      *options.graph = std::move(graph_out);
      return Status::OK();
  }
}  // namespace tensorflow
