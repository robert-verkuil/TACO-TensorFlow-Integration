#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <iostream>
#include "include/taco/tensor.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output(0) = input(0);

    taco::Format dv({taco::Dense});
    taco::Format csr({taco::Dense,taco::Sparse});
    taco::Format csf({taco::Sparse,taco::Sparse,taco::Sparse});
    taco::Format  sv({taco::Sparse});

    // Create tensor, in the right shape! (TODO)s
    taco::Tensor<double> A({2},   dv);
    taco::Tensor<double> B({2,3}, csr);
    taco::Tensor<double> c({3},   dv);

    // Load the data in properly
    

    // Insert data into B and c
//    B.insert({0,0}, 1.0);
//    B.insert({1,2}, 2.0);
//    B.insert({1,3}, 3.0);
//    c.insert(0, 4.0);
//    c.insert(1, 5.0);

    // Pack data as described by the formats
    B.pack();
    c.pack();

    // Form a tensor-vector multiplication expression
    taco::Var i, j(taco::Var::Sum);
    A(i) = B(i,j) * c(j);

    // Compile the expression
    A.compile();

    // Assemble A's indices and numerically compute the result
    A.assemble();
    A.compute();

    std::cout << A << std::endl;
  }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
