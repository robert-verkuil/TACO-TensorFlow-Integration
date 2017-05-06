/// \file inner_product_grad.cca
/// \author David Stutz
/// \brief Implementation of the gradient of a inner product operation, see
/// inner_product.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// the gradients are simply passed as additional arguments as
// they are available in the Python function for registering the gradient operation.
REGISTER_OP("MyMatmulGrad")
  .Input("grad: double")
  .Input("weights: double")
  .Input("input: double")
  .Output("grad_weights: double")
  .Output("grad_input: double");

/// \brief Implementation of an inner product gradient operation.
/// Note that this operation is used in Python to register the gradient as
/// this is not possible in C*+ right now.
/// \param context
/// \author David Stutz
class MyMatmulGradOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit MyMatmulGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  /// \brief Compute the inner product gradients.
  /// \param context
  void Compute(OpKernelContext* context) override {
    
    // output and grad is provided as input
    DCHECK_EQ(3, context->num_inputs());

    // get the gradient tensor
    const Tensor& grad = context->input(0);
    
    // get the original input tensor
    const Tensor& input = context->input(2);
    
    // get the weight tensor
    const Tensor& weights = context->input(1);
    
    // create input shape (inferred from the additional attribute `n`)
    TensorShape input_shape = input.shape();
    TensorShape weights_shape = weights.shape();
    
    DCHECK_EQ(input_shape.dim_size(0), weights_shape.dim_size(1));
    DCHECK_EQ(weights_shape.dim_size(0), grad.shape().dim_size(0));
    
    // create output tensors
    Tensor* grad_input = NULL;
    Tensor* grad_weights = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, input_shape, &grad_input));
    OP_REQUIRES_OK(context, context->allocate_output(0, weights_shape, &grad_weights));
    
    // get the Eigen tensors for data access
    auto grad_tensor = grad.matrix<double>();
    auto weights_tensor = weights.matrix<double>();
    auto input_tensor = input.matrix<double>();
    auto grad_input_tensor = grad_input->matrix<double>();
    auto grad_weights_tensor = grad_weights->matrix<double>();
    
    // std::cout << "\ninput_tensor1: " << input_tensor/*.format(fmt)*/ << std::endl;
    // doign it manually for ismplicity
    for (int i = 0; i < input_shape.dim_size(0); i++) {
      grad_input_tensor(i, 0) = 0;
      for (int j = 0; j < grad.shape().dim_size(0); j++) {
        grad_input_tensor(i, 0) += grad_tensor(j, 0)*weights_tensor(j, i);
        // std::cout << "i: " << i << ",j: " << j << "\ninput_tensor1: " << input_tensor/*.format(fmt)*/ << std::endl;
      }
    }
    
    // std::cout << "\ninput_tensor2: " << input_tensor/*.format(fmt)*/ << std::endl;
    // print the grad_tensor and the input_tensor
    // const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, "\t", " ", "", "");

    for (int i = 0; i < weights_shape.dim_size(0); i++) {
      for (int j = 0; j < weights_shape.dim_size(1); j++) {
        grad_weights_tensor(i, j) = grad_tensor(i, 0)*input_tensor(j, 0);;
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MyMatmulGrad").Device(DEVICE_CPU), MyMatmulGradOp);
