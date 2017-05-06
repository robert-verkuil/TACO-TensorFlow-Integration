#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <iostream>
#include "include/taco/tensor.h"
// #include <stdlib.h>

using namespace tensorflow;

REGISTER_OP("MyMatmul")
    .Input("sparse_m: double")
    .Input("v: double")
    .Output("result: double")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .SetShapeFn(::tensorflow::shape_inference::MatMulShape);

class MyMatmulOp : public OpKernel {
 public:
  explicit MyMatmulOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor_a = context->input(0);
    //auto sparse_m = input_tensor_a.flat<int32>();
    auto sparse_m = input_tensor_a.matrix<double>();
    const Tensor& input_tensor_b = context->input(1);
    auto v = input_tensor_b.flat<double>();
    //auto v = input_tensor_b;

    // Create an output tensor
    Tensor* output_tensor = NULL;

    const TensorShape& weights_shape = input_tensor_a.shape();
    TensorShape output_shape;
    output_shape.AddDim(weights_shape.dim_size(0));
    output_shape.AddDim(1);
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
    auto output = output_tensor->flat<double>();

//////////////////////////////////////////////////// TACO STUFF //////////////////////////////////

    taco::Format dv({taco::Dense});
    taco::Format csr({taco::Dense,taco::Sparse});
    taco::Format csf({taco::Sparse,taco::Sparse,taco::Sparse});
    taco::Format sv({taco::Sparse});

    // Create tensor, in the right shape!
    taco::Tensor<double> a({(int)input_tensor_a.shape().dim_size(0)},   dv);
    taco::Tensor<double> B({(int)input_tensor_a.shape().dim_size(0),
                           (int)input_tensor_a.shape().dim_size(1)}, csr);
    taco::Tensor<double> c({(int)v.size()},   dv);

    // Load the data in properly
//    const int N = dv.size();
    for (int r = 0; r < input_tensor_a.shape().dim_size(0); r++) {
      for (int c = 0; c < input_tensor_a.shape().dim_size(1); c++) {
        if (sparse_m(r, c) != 0) {
          B.insert({r, c}, sparse_m(r, c));
          // std::cout << "inserted " << sparse_m(r, c) << " into B at " << r << ", " << c << std::endl;
        }
      }
    }

    const int N = v.size();
    for (int i = 0; i < N; i++) {
      if (v(i) != 0) {
        c.insert(i, v(i));
        // std::cout << "inserted " << v(i) << " into c at " << i << std::endl;
      }
    }

    // print out the fmts
    // std::cout << "Formats= ";
    // for (auto ds : {a.getFormat(), B.getFormat(), c.getFormat()}) {
        // std::cout << "(" << taco::util::join(ds) << "), ";
    // }

    // print out the dimenions
    // std::cout << "Dimensions= ";
    for (auto ds : {a.getDimensions(), B.getDimensions(), c.getDimensions()}) {
        // std::cout << "(" << taco::util::join(ds) << "), ";
    }

    // // // print out all possible information
    // std::cout << "\nAll info= ";
    // for (auto t : {a, B, c}) {
        // std::cout << t << std::endl;
        // std::cout << "Name: (" << taco::util::join(t.getName()) << "), ";
        // std::cout << "Order: (" << t.getOrder() << "), ";
        // std::cout << "Dimenions: (" << taco::util::join(t.getDimensions()) << "), ";
        // std::cout << "Format: (" << t.getFormat() << "), ";
        // std::cout << "Storage: (" << t.getStorage() << "), ";
        // std::cout << "IndexVars: (" << taco::util::join(t.getIndexVars()) << "), ";
        // std::cout << "Expr: (" << t.getExpr() << "), ";
        // std::cout << "ComponentType: (" << t.getComponentType() << "), ";
    // }

    // Pack data as described by the formats
    // std::cout << "packing " << std::endl;
    B.pack();
    c.pack();

    // Form a tensor-vector multiplication expression
    // std::cout << "setting up expression" << std::endl;
    taco::Var i, j(taco::Var::Sum);
    a(i) = B(i,j) * c(j);

    // Compile the expression
    // std::cout << "compiling" << std::endl;
    a.compile();

    // Assemble A's indices and numerically compute the result
    // std::cout << "assembling" << std::endl;
    a.assemble();
    // std::cout << "computing" << std::endl;
    a.compute();

    // std::cout << a << std::endl;
   
    double* output_vals = a.getStorage().getValues();
    // size_t nvals = a.getStorage().getSize().values;
    size_t nvals = weights_shape.dim_size(0);
    for (int i = 0; i < nvals; i++) {
      output(i) = output_vals[i];
      // std::cout << "inserted " << output_vals[i] << " into output at " << i << std::endl;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MyMatmul").Device(DEVICE_CPU), MyMatmulOp);
