/**
 * ACA Project
 * Definition of a new OP that will the fusion of other two elementary ops
 * 
 * We are defining the fusion of "Add" and "Mul"
 * output = input1 + (input2 * input3) //can also be interpeted as a linear equation Y = (m*X) + b
 **/


#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace tensorflow {
namespace {

class LinearEqOp : public XlaOpKernel {
 public:
  explicit LinearEqOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    if (!ctx->ValidateInputsAreSameShape(this)) return;

    OP_REQUIRES(ctx, ctx->num_inputs() >= 1,
                errors::InvalidArgument("AddN requires at least one argument"));

    xla::ComputationDataHandle sum = ctx->Input(0);
    for (int i = 1; i < ctx->num_inputs(); ++i) {
      sum = ctx->builder()->Add(sum, ctx->Input(i));
    }

    ctx->SetOutput(0, sum);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(LinearEqOp);
};

REGISTER_XLA_OP(Name("LinearEq").CompilationOnly(), LinearEqOp);

}  // namespace
}  // namespace tensorflow