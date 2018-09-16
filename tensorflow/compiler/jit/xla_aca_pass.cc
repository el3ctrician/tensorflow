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
#include <string>
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

    const Edge* edges[10];
    const Edge* subedges[10]; //store all subedges of the edges of the Add operation
    Node* add_node;     //store the add node
    bool found_addmulops = false;
    bool compile_with_xla = false;
    bool compile = false;
    
    // Loop through our graph nodes !.
    for (Node* n : graph_out->op_nodes()) {
      // VLOG(1) << "ACA_Project : -------------------------Node Analysis---------------------------";
      // VLOG(1) << "ACA_Project : node op is : " << n->type_string();
      // //VLOG(1) << "ACA_Project : node summary :" << SummarizeNode(*n);
      // VLOG(1) << "ACA_Project : node num_inputs :" << n->num_inputs();
      
      //Find an Add Operation
      if(n->name() == "Add"){
        VLOG(1) << "ACA_Project : Found possibile optimization candidate ";
        VLOG(1) << "ACA_Project : -------------------------Node Analysis---------------------------";
        
        Status status = GetNodeAttr(n->attrs(), kXlaCompileAttr, &compile);
        if (status.ok() && compile) {
          VLOG(1) << "ACA_Project : compile with XLA --> Optimize";
          compile_with_xla = true;
        }
        else  
          VLOG(1) << "ACA_Project : No XLA, SKIP THIS OPTIMIZATION PASS!!";

        VLOG(1) << "ACA_Project : node op is : " << n->type_string();
        VLOG(1) << "ACA_Project : node num_inputs :" << n->num_inputs();
        //VLOG(1) << "ACA_Project : node shape :";// << n->shape().DebugString();
        //VLOG(1) << "ACA_Project : node summary :" << SummarizeNode(*n);

        VLOG(1) << "ACA_Project : Node Input Edges : ";
        int i=0;
        add_node = n; //store the add node
        // Loop through the output edges
        for (const Node* node : add_node->out_nodes()) {
          VLOG(1) << "    +ACA_Project : output node/edge op is : " << node->type_string();// << " - shape: " << node->shape().DebugString(); 
        }
        // Loop through the input edges
        for (const Edge* edge : n->in_edges()) {
          VLOG(1) << "    ACA_Project : input node/edge op is : " << edge->src()->type_string();// << " - shape: " << edge->src()->shape().DebugString();
          edges[i++] = edge; //store all the edges of the Add operation
    
          if(edge->src()->type_string() == "MatMul"){
              VLOG(1) << "      ACA_Project : Node MatMul Analysis";
              found_addmulops = true;
              int j=0;
              // Loop through the input edges of the edges
              for (const Edge* subedge : edge->src()->in_edges()){
                VLOG(1) << "          ACA_Project : input node/edge op is : " << subedge->src()->type_string();// << " - shape: " << subedge->src()->shape().DebugString();
                subedges[j++] = subedge;
              }
          }              
        }
        VLOG(1) << "ACA_Project : -------------------------END Node Analysis---------------------------";
      } //end compilation candiate Analysiss
    } //end graph loop


    //Optimization
    if(found_addmulops && compile_with_xla){
      //New node creation
      VLOG(1) << "ACA_Project : Starting node substition";
      Status status;
      NodeDef node_def;// = add_node->def();
      node_def.set_name("LinearEqOp_optimized");
      node_def.set_op("LinearEq");
      //AddNodeAttr( "LinearEq", 0, &node_def);
      Node* new_node = graph_out->AddNode(node_def, &status);
      string tmp = new_node->type_string();

      //Bisogna aggiungere un edge che collega l'output del nuovo nuovo con il nodo che riceveva in ingresso Add
      Node* top_node;
      int top_node_index;
      for (Node* n : add_node->out_nodes()){
          top_node = n; //take the first output
          break;        //and leave
      }
      for (const Edge* top_edge :top_node->in_edges()){
        if (top_edge->src()->type_string() == "Add"){
          top_node_index = top_edge->dst_input();
        }
      }
      graph_out->AddEdge(new_node, 0, top_node, top_node_index);
      //(Node* source, int x, Node* dest, int y) 

      //Modify the graph
      //Connect the inputs of the MatMul operation to the new operation
      graph_out->AddEdge(subedges[0]->src(), 0, new_node, 0);
      graph_out->AddEdge(subedges[1]->src(), 0, new_node, 1);
      //Bisogna aggiungere il secondo input del nodo principale
      graph_out->AddEdge(edges[1]->src(), 0, new_node, 2);

      //remove node and edge after setted up the new node    
      graph_out->RemoveEdge(edges[0]);    //remove MatMul node
      graph_out->RemoveNode(add_node);    //remove Add node
    }

    if(compile_with_xla){
      //Print again everything so that we can verify
      VLOG(1) << "ACA_Project : Verify Node substition";
      // Loop through our graph nodes !.
      for (Node* n : graph_out->op_nodes()) {
        // VLOG(1) << "ACA_Project : +++++++++++++Node Analysis+++++++++++++";
        // VLOG(1) << "ACA_Project : node op is : " << n->type_string();
        // VLOG(1) << "ACA_Project : node num_inputs :" << n->num_inputs();
        // VLOG(1) << "ACA_Project : Node name : " << n->name();
  
        //Find an LinearEq Operation
        if(n->type_string() == "LinearEq"){
          VLOG(1) << "ACA_Project : +++++++++++++Substited Node Analysis+++++++++++++";

          // Loop through the input edges
          for (const Edge* edge : n->in_edges()) {
            VLOG(1) << "    ACA_Project : input node/edge op is : " << edge->src()->type_string();
          }
          // Loop through the output edges
          for (const Node* node : n->out_nodes()) {
            VLOG(1) << "    +ACA_Project : output node/edge op is : " << node->type_string(); 
          }
          VLOG(1) << "ACA_Project : +++++++++++++END Node Input Edges Analysis+++++++++++++";
        }

        // VLOG(1) << "ACA_Project : +++++++++++++End Substited Node Analysis+++++++++++++";
      }
      // VLOG(1) << "ACA_Project : ++++++++++++++++END++++++++++++++++++++++++++";
    }

      //update the graph
      //*options.graph = std::move(graph_out);
      return Status::OK();
  }
}  // namespace tensorflow
