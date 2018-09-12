/**
 * ACA Project
 * Definition of a new OP that will the fusion of other two elementary ops
 * 
 * We are defining the fusion of "Add" and "Mul"
 * output = input1 + (input2 * input3) // can also be understood as linear eq Y = mX + b
 **/


/**
 * The following code is for a normal Tensorflow op and rappresents a high level implementation
 * XLA code is slightly diffrent and Ops can be found in tensorflow/compiler/tf2xla/kernels
 * this is why this code will be moved there and ops will be implemented accordingly
 * Code is left here for reference 
 *  */
/** 
 * NB : add this to BUILD to build this file
 *                   |
 *                   |
 *                   v
 * cc_library(
    name = "xla_aca_ops",
    srcs = ["xla_aca_ops.cc"],
)
**/


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"


using namespace tensorflow;

REGISTER_OP("LinearEq")
    .Input("b: int32")
    .Input("x : int32")
    .Input("m : int32")
    .Output("y: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });


class LinearEqOp : public OpKernel {
 public:
  explicit LinearEqOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    VLOG(1) << "ACA_Project : LinearEqOp statrt in core/kernels";

    // Grab the input tensor
    const Tensor& input_mul1_tensor = context->input(0);
    const Tensor& input_mul2_tensor = context->input(1);
    const Tensor& input_add_tensor = context->input(2);
    auto input_mul1 = input_mul1_tensor.flat<int32>();
    auto input_mul2 = input_mul2_tensor.flat<int32>();
    auto input_add = input_add_tensor.flat<int32>();

    // Create an output tensor
    Tensor output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_add_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    /*output_tensor = context->Add(output_tensor, input_mul1_tensor);
    output_tensor = context->Mul(output_tensor, input_mul2_tensor);
    output_tensor = context->Add(output_tensor, input_add_tensor);*/

    const int N = input_add.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = input_add(i) + (input_mul1(i) * input_mul2(i));
    }

    //Set the output tensor
    context->set_output(0, &output_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("LinearEq").Device(DEVICE_CPU), LinearEqOp);


