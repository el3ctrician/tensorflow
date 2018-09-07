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

    // Loop through our graph nodes !.
    for (Node* n : graph_out->op_nodes()) {
      VLOG(1) << "ACA_Project : -------------------------Node Analysis---------------------------";
      VLOG(1) << "ACA_Project : node op is : " << n->type_string();
      //VLOG(1) << "ACA_Project : node summary :" << SummarizeNode(*n);
      VLOG(1) << "ACA_Project : node num_inputs :" << n->num_inputs();
      
      //Find an Add Operation
      if(n->name() == "Add"){
        VLOG(1) << "ACA_Project : -------------------------Node Input Edges Analysis---------------------------";

        int i=0;
        add_node = n; //store the add node

        // Loop through the output edges
        for (const Edge* edge : add_node->out_edges()) {
          VLOG(1) << "    +ACA_Project : output node/edge op is : " << edge->src()->type_string(); 
        }
        // Loop through the input edges
        for (const Edge* edge : n->in_edges()) {
          VLOG(1) << "    ACA_Project : input node/edge op is : " << edge->src()->type_string();
          edges[i++] = edge; //store all the edges of the Add operation

          if(edge->src()->type_string() == "MatMul"){
              VLOG(1) << "      ACA_Project : -------------------------Node Input Edge of an Edge Analysis---------------------------";

              found_addmulops = true;
              int j=0;
              // Loop through the input edges of the edges
              for (const Edge* subedge : edge->src()->in_edges()){
                VLOG(1) << "          ACA_Project : input node/edge op is : " << subedge->src()->type_string();
                subedges[j++] = subedge;

                //Connect the inputs of the MatMul operation to the new operation
                //graph_out->AddEdge(subedge->src(), subedge->dst_input(), new_node, i++);
              }

              //Create a new operation that has 2 of the "MatMul" node inputs, the other 
              //input of "Add" node and it's output
              //It would be a problem if more than one edge of Add where to be a MatMul operation

              //### bisogna settare l'operazione da qualche parte, forse con node_def!!
              //graph_out->AddEdge(subedges[0]->src(), subedges[0]->dst_input(), new_node, 0);
              //graph_out->AddEdge(subedges[1]->src(), subedges[1]->dst_input(), new_node, 1);

              VLOG(1) << "      ACA_Project : -------------------------END Node Input Edge of an Edge Analysis---------------------------";
          }
          else{
              //Bisogna aggiungere il secondo input del nodo principale
              //graph_out->AddEdge(edge->src(), edge->dst_input(), new_node, i++);
          }
              
        }

        //remove node and edge after setted up the new node
        //graph_out->RemoveNode(n);               
        //graph_out->RemoveEdge(edges[0]);

        VLOG(1) << "ACA_Project : -------------------------END Node Input Edges Analysis---------------------------";
      }
      VLOG(1) << "ACA_Project : ------------------------End Node Analysis--------------------------";
    }
    VLOG(1) << "ACA_Project : -----------------------------END---------------------------------";


    if(found_addmulops){
      //New node creation
      Status status;
      NodeDef node_def;
      //node_def.set_name(graph_out->NewName("LinearEqOp"));
      node_def.set_op("LinearEq");
      //AddNodeAttr( "LinearEq", 0, &node_def);
      Node* new_node = graph_out->AddNode(node_def, &status);
      string tmp = new_node->type_string();

      //Bisogna aggiungere un edge che collega l'output del nuovo nuovo con il nodo che riceveva in ingresso Add
      Node* top_node;
      for (Node* n : add_node->out_nodes()){
          top_node = n; //take the first output
          break;        //and leave
      }
      graph_out->AddEdge(new_node, 0, top_node, 0);
      //(Node* source, int x, Node* dest, int y) 

      //Modify the graph
      //Connect the inputs of the MatMul operation to the new operation
      graph_out->AddEdge(subedges[0]->src(), subedges[0]->dst_input(), new_node, 0);
      graph_out->AddEdge(subedges[1]->src(), subedges[1]->dst_input(), new_node, 1);
      //Bisogna aggiungere il secondo input del nodo principale
      graph_out->AddEdge(edges[1]->src(), edges[1]->dst_input(), new_node, 2);

      //remove node and edge after setted up the new node    
      graph_out->RemoveEdge(edges[0]);    //remove MatMul node
      graph_out->RemoveNode(add_node);    //remove Add node
    }


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
      //Find an LinearEq Operation
      else if(n->type_string() == "LinearEq"){
        VLOG(1) << "ACA_Project : +++++++++++++Node Input Edges Analysis+++++++++++++";

        // Loop through the input edges
        for (const Edge* edge : n->in_edges()) {
          VLOG(1) << "    ACA_Project : input node/edge op is : " << edge->src()->type_string();
        }
        VLOG(1) << "ACA_Project : +++++++++++++END Node Input Edges Analysis+++++++++++++";
      }

      VLOG(1) << "ACA_Project : +++++++++++++End Node Analysis+++++++++++++";
    }
    VLOG(1) << "ACA_Project : ++++++++++++++++END++++++++++++++++++++++++++";


      //update the graph
      //*options.graph = std::move(graph_out);
      return Status::OK();
  }
}  // namespace tensorflow
