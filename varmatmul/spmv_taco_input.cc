#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <iostream>
#include "../include/taco/tensor.h"

#include "../include/taco/util/error.h"
#include "../include/taco/util/strings.h"
#include "../include/taco/util/timers.h"
#include "../include/taco/util/fill.h"
#include "../include/taco/util/env.h"
#include "../include/taco/util/collections.h"

#include "../include/taco/io/tns_file_format.h"

using namespace tensorflow;

/*

For sparse_fmt parameter
0   =   dd <- default
1   =   ds 
2   =   sd
3   =   ss

*/

// If tns_file == "" then use this op for SpMV.
// Otherwise load the tns from filename and multiply by a 
// randomly taco-generated vector.

// namespace shape_inference {
REGISTER_OP("SpmvTacoInput")
    .Input("level_index0: int32")
    .Input("level_index1: int32")
    .Input("values: double")
    .Input("sizes: int32")

    .Input("v: double")


    .Attr("sparse_fmt: list(bool) = [true, true]")
    .Attr("tns_file: string = \"\"")
    .Attr("v_fmt_str: string = \"\"")
    .Output("result: double")
    .Output("times: double")


    // .Output("product: double")
    .Attr("adjoint_a: bool = false")
    .Attr("adjoint_b: bool = false")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // ::tensorflow::shape_inference::DimensionHandle unused_dim;
      // ::tensorflow::shape_inference::ShapeHandle unused;
      // ::tensorflow::shape_inference::ShapeHandle b;
      // ::tensorflow::shape_inference::ShapeHandle a_shape;
      // ::tensorflow::shape_inference::ShapeHandle a_shape_shape;
      // TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));  // a_indices
      // TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));  // a_values
      // TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &a_shape));
      // TF_RETURN_IF_ERROR(c->WithRank(a_shape, 2, &a_shape));
      // TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &b));

      // bool adjoint_a;
      // bool adjoint_b;
      // TF_RETURN_IF_ERROR(c->GetAttr("adjoint_a", &adjoint_a));
      // TF_RETURN_IF_ERROR(c->GetAttr("adjoint_b", &adjoint_b));

      // ::tensorflow::shape_inference::DimensionHandle output_right = c->Dim(b, adjoint_b ? 0 : 1);
      // ::tensorflow::shape_inference::DimensionHandle output_left = c->Dim(a_shape, adjoint_a ? 1 : 0);
      // ::tensorflow::shape_inference::DimensionHandle inner_left = c->Dim(a_shape, adjoint_a ? 0 : 1);
      // ::tensorflow::shape_inference::DimensionHandle inner_right = c->Dim(b, adjoint_b ? 1 : 0);
      // TF_RETURN_IF_ERROR(c->Merge(inner_left, inner_right, &unused_dim));
      // c->set_output(0, c->Matrix(output_left, output_right));

      // // c->set_output(0, c->Matrix(0, 100));
      // c->set_output(1, c->Matrix(1, 100));
      return Status::OK();
    }); // taken form tensorflow/core/ops/sparse_ops.cc
// }

class SpmvTacoInputOp : public OpKernel {
 private:
    std::vector<bool> sparse_fmt_;
    std::string tns_file_;
    std::string v_fmt_str_;


 public:
  explicit SpmvTacoInputOp(OpKernelConstruction* context) : OpKernel(context) {
    // Grab the sparse_fmt attribute
    OP_REQUIRES_OK(context,
                   context->GetAttr("sparse_fmt", &sparse_fmt_));
    // Check that sparse_fmt is in allowable range
    OP_REQUIRES(context, sparse_fmt_.size() > 0,
            errors::InvalidArgument("Need preserve_index >= 0, got ",
                                    sparse_fmt_.size()));
    // Load the tns file name
    OP_REQUIRES_OK(context,
               context->GetAttr("tns_file", &tns_file_));
    // Load the vector format string
    OP_REQUIRES_OK(context,
               context->GetAttr("v_fmt_str", &v_fmt_str_));
  }

  // acceptable fill methods are
  // d, u, r, s, h, v, l, f, b
  taco::util::FillMethod strToFill(std::string fmt_str) {
    taco::util::FillMethod ret;
    switch (fmt_str[0]) {
        case 'd': {
          ret = taco::util::FillMethod::Dense;
          break;
        }
        case 'u': {
          ret = taco::util::FillMethod::Uniform;
          break;
        }
        case 'r': {
          ret = taco::util::FillMethod::Random;
          break;
        }
        case 's': {
          ret = taco::util::FillMethod::Sparse;
          break;
        }
        case 'h': {
          ret = taco::util::FillMethod::HyperSpace;
          break;
        }
        case 'v': {
          ret = taco::util::FillMethod::SlicingV;
          break;
        }
        case 'l': {
          ret = taco::util::FillMethod::SlicingH;
          break;
        }
        case 'f': {
          ret = taco::util::FillMethod::FEM;
          break;
        }
        case 'b': {
          ret = taco::util::FillMethod::Blocked;
          break;
        }
        default: {
          ret = taco::util::FillMethod::Dense;
          break;
        }
    }
    return ret;
  }

  void Compute(OpKernelContext* context) override {
    taco::util::TimeResults tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7;
    // Grab the input tensor
    taco::util::Timer timer0;
    timer0.start();
            // const Tensor& input_tensor_a = context->input(0);
            // auto sparse_m = input_tensor_a.matrix<double>();
            // const Tensor& input_tensor_b = context->input(1);
            // auto v = input_tensor_b.flat<double>();

    // Create an output tensor
    const Tensor* level_index0_;
    const Tensor* level_index1_;
    const Tensor* values_;
    const Tensor* sizes_;
    const Tensor* v_;
    OP_REQUIRES_OK(context, context->input("level_index0", &level_index0_));
    OP_REQUIRES_OK(context, context->input("level_index1", &level_index1_));
    OP_REQUIRES_OK(context, context->input("values", &values_));
    OP_REQUIRES_OK(context, context->input("sizes", &sizes_));
    OP_REQUIRES_OK(context, context->input("v", &v_));
    auto level_index0 = level_index0_->flat<int>();
    auto level_index1 = level_index1_->flat<int>();
    auto values = values_->flat<double>();
    auto sizes = sizes_->flat<int>();
    auto v = v_->flat<double>();

    const int* level_index0_raw = level_index0.data();
    const int* level_index1_raw = level_index1.data();
    const double* values_raw = values.data();
    // const int* sizes_raw = sizes.data();
    // const double* v_raw = v.data();


    Tensor* output_tensor = NULL;
    TensorShape output_shape;
    output_shape.AddDim(v.size());
    output_shape.AddDim(1);
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
    auto output = output_tensor->flat<double>();
    timer0.stop();
    tr0 = timer0.getResult();



    // Create an output tensor of gathered times
    Tensor* time_output_tensor = NULL;
    TensorShape time_output_shape;
    time_output_shape.AddDim(10);
    time_output_shape.AddDim(1);
    OP_REQUIRES_OK(context, context->allocate_output(1, time_output_shape,
                                                 &time_output_tensor));
    auto time_output = time_output_tensor->flat<double>();



    // ////////////////////////////// TACO STUFF //////////////////////////////////
    // Make the needed formats
    std::vector<taco::LevelType> fmts_list;
    for (int f : sparse_fmt_) {
        if (f==0) {
            fmts_list.push_back(taco::Sparse);
        } else {
            fmts_list.push_back(taco::Dense);
        }
    }
    taco::Format sp_mtx_fmt_to_use(fmts_list);
    taco::Format dv({taco::Dense});
    taco::Format CSR({taco::Dense, taco::Sparse},{0,1});
    taco::Format CSC({taco::Dense, taco::Sparse},{1,0});


    taco::Tensor<double> result_tensor;


    // Make the taco objects
    taco::util::Timer timer1;
    timer1.start();
    taco::Tensor<double> a({(int)v.size()},   dv);
    // taco::Tensor<double> B({10,10}, sp_mtx_fmt_to_use);
    taco::Tensor<double> B({(int)v.size(),(int)v.size()}, CSR);
    taco::Tensor<double> c({(int)v.size()},   dv);
    timer1.stop();
    tr1 = timer1.getResult();



    // // FOR SPMV USAGE
    taco::util::Timer timer2;
    timer2.start();


    // std::cout << "before setCSR really " << std::endl;


    B.setCSR((double*)values_raw, (int *)level_index0_raw, (int *)level_index1_raw);
    // std::cout << "after setCSR" << B << "done printing" << std::endl;

    // std::cout << B.getStorage().getLevelIndex(0).ptr << ", " << B.getStorage().getLevelIndex(0).idx << std::endl;
    // std::cout << B.getStorage().getLevelIndex(1).ptr << ", " << B.getStorage().getLevelIndex(1).idx << std::endl;

    // // // Populate the vector
    const int N = v.size();
    for (int i = 0; i < N; i++) {
      if (v(i) != 0) {
        c.insert(i, v(i));
      }
    }

    timer2.stop();
    tr2 = timer2.getResult();

    // // Pack data as described by the formats
    // std::cout << "packing B" << std::endl;
    // TACO_TIME_REPEAT(B.pack(), 1, tr3)
    // std::cout << "packing c" << std::endl;
    TACO_TIME_REPEAT(c.pack(), 1, tr4)

    // // Form a tensor-vector multiplication expression
    // std::cout << "setting up index expression" << std::endl;
    taco::Var i, j(taco::Var::Sum);
    a(i) = B(i,j) * c(j);

    result_tensor = a;



    // // Compile the expression
    // std::cout << "compiling" << std::endl;
    TACO_TIME_REPEAT(result_tensor.compile(), 1, tr5)
    // // std::cout << "compile took " << tr3 << std::endl;

    // // Assemble A's indices and numerically compute the result
    // std::cout << "assembling" << std::endl;
    TACO_TIME_REPEAT(result_tensor.assemble(), 1, tr6)
    // // std::cout << "assemble took " << tr4 << std::endl;

    // std::cout << "computing" << std::endl;
    TACO_TIME_REPEAT(result_tensor.compute(), 1, tr7)
    // std::cout << "compute took " << tr5 << std::endl;

    double* output_vals = result_tensor.getStorage().getValues();
    size_t nvals = v.size();
    for (int i = 0; i < nvals; i++) {
      output(i) = output_vals[i];
    }

    int counter = 0;
    for (auto tr : {tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7}) {
        // std::cout << tr.mean << ", ";
        time_output(counter++) = tr.mean;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("SpmvTacoInput").Device(DEVICE_CPU), SpmvTacoInputOp);
