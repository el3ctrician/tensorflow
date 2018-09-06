/**
 * ACA Project
 * Definition of a new OP that will the fusion of other two elementary ops
 * 
 * We are defining the fusion of "Add" and "Mul"
 * output = input1 + (input2 * input3) //can also be interpeted as a linear equation Y = (m*X) + b
 **/

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace tensorflow {
namespace {

class LinearEqOp : public XlaOpKernel {
 public:
  explicit LinearEqOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    //Function to be implemented 
    if (!ctx->ValidateInputsAreSameShape(this)) return;

    OP_REQUIRES(ctx, ctx->num_inputs() != 3,
                errors::InvalidArgument("LinearEq requires exactly 3 arguments!"));

    xla::ComputationDataHandle sum = ctx->Input(0);
    sum = ctx->builder()->Mul(sum, ctx->Input(1));    //Multiplied input0 to input1: input0 * input1
    sum = ctx->builder()->Add(sum, ctx->Input(2));    //Sumed input2 to the rest: input0 * input1 + input2

    //Set the output
    ctx->SetOutput(0, sum);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(LinearEqOp);
};

REGISTER_XLA_OP(Name("LinearEq").AllowResourceTypes(), LinearEqOp);

}  // namespace
}  // namespace tensorflow