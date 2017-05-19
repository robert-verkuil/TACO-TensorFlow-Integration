#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <iostream>
// #include "include/taco/tensor.h"

using namespace tensorflow;

REGISTER_OP("Noop")
    .Input("sparse_m: double")
    .Input("v: double")
    .Output("result: double")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .SetShapeFn(::tensorflow::shape_inference::MatMulShape);

class NoopOp : public OpKernel {
 public:
  explicit NoopOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // // Grab the input tensor
    // const Tensor& input_tensor_a = context->input(0);
    // auto sparse_m = input_tensor_a.matrix<double>();
    // const Tensor& input_tensor_b = context->input(1);
    // auto v = input_tensor_b.flat<double>();

    // // Create an output tensor
    Tensor* output_tensor = NULL;

    // const TensorShape& weights_shape = input_tensor_a.shape();
    TensorShape output_shape;
    // output_shape.AddDim(weights_shape.dim_size(0));
    output_shape.AddDim(1);
    output_shape.AddDim(1);
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
    // auto output = output_tensor->flat<double>();
  }
};

REGISTER_KERNEL_BUILDER(Name("Noop").Device(DEVICE_CPU), NoopOp);
