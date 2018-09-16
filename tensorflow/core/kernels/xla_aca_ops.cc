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

//maybe no need for this
REGISTER_OP("LinearEq")
    .Input("a: float")
    .Input("b : float")
    .Input("c : float")
    .Output("y: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

template <typename Device, typename T, bool USE_CUBLAS>
class LinearEqOp : public OpKernel {
 public:
  explicit LinearEqOp(OpKernelConstruction* context) : OpKernel(context) {
   // OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
   // OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b_));
  }

  void Compute(OpKernelContext* context) override {
    VLOG(1) << "ACA_Project : LinearEqOp starts in core/kernels";

    // Grab the input tensor
    const Tensor& input_mul1_tensor = context->input(0);
    const Tensor& input_mul2_tensor = context->input(1);
    const Tensor& input_add_tensor = context->input(2);
    auto input_mul1 = input_mul1_tensor.flat<float>();
    auto input_mul2 = input_mul2_tensor.flat<float>();
    auto input_add = input_add_tensor.flat<float>(); //remove casting

    // Check that the dimensions of the three matrices are valid.
    /*OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_mul1_tensor.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_mul2_tensor.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_add_tensor.shape()),
                errors::InvalidArgument("In[2] is not a matrix"));*/
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    // Create an output tensor
    OP_REQUIRES(
        context, input_mul1_tensor.dim_size(dim_pair[0].first) == input_mul2_tensor.dim_size(dim_pair[0].second),
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", input_mul1_tensor.shape().DebugString(),
            ", In[1]: ", input_mul2_tensor.shape().DebugString()));
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape({input_mul1_tensor.dim_size(a_dim_remaining), input_mul2_tensor.dim_size(b_dim_remaining)});
    Tensor* out = nullptr;

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    if (input_mul1_tensor.NumElements() == 0 || input_mul2_tensor.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      SetZeroFunctor<Device, T> f;
      f(context->eigen_device<Device>(), out->flat<T>());
      return;
    }

    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    auto output_flat = out->flat<float>();

    //Do the operation
    const int N = input_add.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = input_add(i) + (input_mul1(i) * input_mul2(i));
    }

    //Set the output tensor
    context->set_output(0, *out);
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
};
REGISTER_KERNEL_BUILDER(Name("LinearEq").Device(DEVICE_CPU), LinearEqOp);



