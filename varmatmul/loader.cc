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
#include "../include/taco/storage/storage.h"

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
REGISTER_OP("RobLoader")
    .Input("a_indices: int64")
    .Input("a_values: double")
    .Input("a_shape: int64")


    .Attr("sparse_fmt: list(bool) = [true, true]")
    .Attr("tns_file: string = \"\"")
    .Attr("v_fmt_str: string = \"\"")

    // .Output("result: double")
    // .Output("times: double")
    .Input("v: double")

    .Output("level_index0: double")
    .Output("level_index1: double")
    .Output("values: double")
    .Output("sizes: double")


    // .Output("product: double")
    .Attr("adjoint_a: bool = false")
    .Attr("adjoint_b: bool = false")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::DimensionHandle unused_dim;
      ::tensorflow::shape_inference::ShapeHandle unused;
      ::tensorflow::shape_inference::ShapeHandle b;
      ::tensorflow::shape_inference::ShapeHandle a_shape;
      ::tensorflow::shape_inference::ShapeHandle a_shape_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));  // a_indices
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));  // a_values
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &a_shape));
      TF_RETURN_IF_ERROR(c->WithRank(a_shape, 2, &a_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &b));

      bool adjoint_a;
      bool adjoint_b;
      TF_RETURN_IF_ERROR(c->GetAttr("adjoint_a", &adjoint_a));
      TF_RETURN_IF_ERROR(c->GetAttr("adjoint_b", &adjoint_b));

      ::tensorflow::shape_inference::DimensionHandle output_right = c->Dim(b, adjoint_b ? 0 : 1);
      ::tensorflow::shape_inference::DimensionHandle output_left = c->Dim(a_shape, adjoint_a ? 1 : 0);
      ::tensorflow::shape_inference::DimensionHandle inner_left = c->Dim(a_shape, adjoint_a ? 0 : 1);
      ::tensorflow::shape_inference::DimensionHandle inner_right = c->Dim(b, adjoint_b ? 1 : 0);
      TF_RETURN_IF_ERROR(c->Merge(inner_left, inner_right, &unused_dim));
      // c->set_output(0, c->Matrix(output_left, output_right));

      // c->set_output(0, c->Matrix(0, 100));
      // c->set_output(1, c->Matrix(1, 100));
      // c->set_output(2, c->Matrix(1, 100));
      // c->set_output(3, c->Matrix(1, 100));
      return Status::OK();
    }); // taken form tensorflow/core/ops/sparse_ops.cc
// }

class RobLoaderOp : public OpKernel {
 private:
    std::vector<bool> sparse_fmt_;
    std::string tns_file_;
    std::string v_fmt_str_;


 public:
  explicit RobLoaderOp(OpKernelConstruction* context) : OpKernel(context) {
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
    const Tensor* a_indices_;
    const Tensor* a_values_;
    const Tensor* a_shape_;
    const Tensor* v_;
    OP_REQUIRES_OK(context, context->input("a_indices", &a_indices_));
    OP_REQUIRES_OK(context, context->input("a_values", &a_values_));
    OP_REQUIRES_OK(context, context->input("a_shape", &a_shape_));
    OP_REQUIRES_OK(context, context->input("v", &v_));
    auto a_indices = a_indices_->matrix<int64>();
    auto a_values = a_values_->flat<double>();
    auto a_shape = a_shape_->flat<int64>();
    auto v = v_->flat<double>();










// Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(v_->shape()),
                errors::InvalidArgument("Tensor 'b' is not a matrix"));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(a_shape_->shape()),
                errors::InvalidArgument("Tensor 'a_shape' is not a vector"));

    OP_REQUIRES(
        context, a_shape_->NumElements() == 2,
        errors::InvalidArgument("Tensor 'a_shape' must have 2 elements"));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(a_values_->shape()),
                errors::InvalidArgument("Tensor 'a_values' is not a vector"));

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(a_indices_->shape()),
                errors::InvalidArgument("Tensor 'a_indices' is not a matrix"));

    OP_REQUIRES(context, a_indices_->shape().dim_size(0) == a_values_->NumElements(),
                errors::InvalidArgument("Number of rows of a_indices does not "
                                        "match number of entries in a_values"));

    OP_REQUIRES(
        context, a_indices_->shape().dim_size(1) == a_shape_->NumElements(),
        errors::InvalidArgument("Number of columns of a_indices does not match "
                                "number of entries in a_shape"));

    auto a_shape_t = a_shape_->vec<int64>();
    const int64 outer_left = a_shape_t(0);
    const int64 outer_right = v_->shape().dim_size(1);
    const int64 inner_left = a_shape_t(1);
    const int64 inner_right = v_->shape().dim_size(0);

    OP_REQUIRES(
        context, inner_right == inner_left,
        errors::InvalidArgument(
            "Cannot multiply A and B because inner dimension does not match: ",
            inner_left, " vs. ", inner_right,
            ".  Did you forget a transpose?  "
            "Dimensions of A: [",
            a_shape_t(0), ", ", a_shape_t(1), ").  Dimensions of B: ",
            v_->shape().DebugString()));

    TensorShape out_shape({outer_left, outer_right});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));









    // Tensor* output_tensor = NULL;
    // TensorShape output_shape;
    // output_shape.AddDim(a_shape(0));
    // output_shape.AddDim(1);
    // OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
    //                                                  &output_tensor));
    // auto output = output_tensor->flat<double>();
    // timer0.stop();
    // tr0 = timer0.getResult();


                // // Create an output tensor of gathered times
                // Tensor* time_output_tensor0 = NULL;
                // TensorShape time_output_shape0;
                // time_output_shape0.AddDim(100);
                // time_output_shape0.AddDim(1);
                // OP_REQUIRES_OK(context, context->allocate_output(0, time_output_shape0,
                //                                              &time_output_tensor0));
                // auto time_output0 = time_output_tensor0->flat<double>();


    // Create an output tensor of gathered times
    // Tensor* time_output_tensor = NULL;
    // TensorShape time_output_shape;
    // time_output_shape.AddDim(100);
    // time_output_shape.AddDim(1);
    // OP_REQUIRES_OK(context, context->allocate_output(1, time_output_shape,
    //                                              &time_output_tensor));
    // auto time_output = time_output_tensor->flat<double>();



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


    taco::Tensor<double> result_tensor;


    // Make the taco objects
    taco::util::Timer timer1;
    timer1.start();
    taco::Tensor<double> a({(int)v.size()},   dv);
    taco::Tensor<double> B({(int)a_shape(0),(int)a_shape(1)}, sp_mtx_fmt_to_use);
    taco::Tensor<double> B2({(int)a_shape(0),(int)a_shape(1)}, sp_mtx_fmt_to_use);
    taco::Tensor<double> c({(int)v.size()},   dv);
    timer1.stop();
    tr1 = timer1.getResult();



    // FOR SPMV USAGE
    taco::util::Timer timer2;
    timer2.start();
    // Populate the sparse matrix
    std::vector<int> tmp;
    for (int r = 0; r < a_indices_->dim_size(0); r++) {
        for (int c = 0; c < a_indices_->dim_size(1); c++) {
          tmp.push_back((int)a_indices(r, c));
          // std::cout << a_indices(r, c) << ", " << (double)a_values(r) << std::endl;
        }
        B.insert(tmp, (double)a_values(r));
        tmp.clear();
    }
    // Populate the vector
    const int N = v.size();
    for (int i = 0; i < N; i++) {
      if (v(i) != 0) {
        c.insert(i, v(i));
      }
    }
    timer2.stop();
    tr2 = timer2.getResult();

    // Pack data as described by the formats
    TACO_TIME_REPEAT(B.pack(), 1, tr3)
    TACO_TIME_REPEAT(c.pack(), 1, tr4)

    // Transfer contents of B -> B2
    // size_t sz = 1000;
    // double* vals = (double*)malloc(sz);
    // int* rowPtr  = (int*)malloc(sz);
    // int* colIdx  = (int*)malloc(sz);
    // double vals[sz];
    // int rowPtr[sz];
    // int colIdx[sz];
    double* vals = NULL;
    int* rowPtr = NULL;
    int* colIdx = NULL;
    B.getCSR(&vals, &rowPtr, &colIdx);
    // std::cout << "middle " << vals << ", " << rowPtr << ", " << colIdx << std::endl;

    // B2.setCSR(vals, rowPtr, colIdx);


    auto S = B.getStorage();
    auto size =  S.getSize();





    Tensor* output_tensor0 = NULL;
    TensorShape output_shape0;
    output_shape0.AddDim(size.indexSizes[1].ptr);
    output_shape0.AddDim(1);
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape0,
                                                     &output_tensor0));
    auto output0 = output_tensor0->flat<double>();

    Tensor* output_tensor1 = NULL;
    TensorShape output_shape1;
    output_shape1.AddDim(size.indexSizes[1].idx);
    output_shape1.AddDim(1);
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape1,
                                                     &output_tensor1));
    auto output1 = output_tensor1->flat<double>();

    Tensor* output_tensor2 = NULL;
    TensorShape output_shape2;
    output_shape2.AddDim(size.values);
    output_shape2.AddDim(1);
    OP_REQUIRES_OK(context, context->allocate_output(2, output_shape2,
                                                     &output_tensor2));
    auto output2 = output_tensor2->flat<double>();

    Tensor* output_tensor3 = NULL;
    TensorShape output_shape3;
    output_shape3.AddDim(3);
    output_shape3.AddDim(1);
    OP_REQUIRES_OK(context, context->allocate_output(3, output_shape3,
                                                     &output_tensor3));
    auto output3 = output_tensor3->flat<double>();

    std::cout << "size.indexSizes[1].ptr" << size.indexSizes[1].ptr << std::endl;
    for (int i = 0; i < size.indexSizes[1].ptr; i++) {
      std::cout << rowPtr[i];
      output0(i) = rowPtr[i];
    }
    std::cout << std::endl;
    std::cout << "size.indexSizes[1].idx" << size.indexSizes[1].idx << std::endl;
    for (int i = 0; i < size.indexSizes[1].idx; i++) {
      std::cout << colIdx[i];
      output1(i) = colIdx[i];
    }
    std::cout << std::endl;
    std::cout << "size.values" << size.values << std::endl;
    for (int i = 0; i < size.values; i++) {
      std::cout << vals[i];
      output2(i) = vals[i];
    }
    std::cout << std::endl;
    output3(0) = size.indexSizes[1].ptr;
    output3(1) = size.indexSizes[1].idx;
    output3(2) = size.values;

      // B2.setCSR(tmp_vals, tmp_rowPtr, tmp_colIdx);
      std::cout << "after copying" << std::endl;

      std::cout << "in loader" << B << std::endl;

    std::cout << "size.values = " << size.values;
    for (taco::storage::TensorStorage::Size::LevelIndexSize levelIndexSize : size.indexSizes) {
      std::cout << "(" << levelIndexSize.ptr << ", " << levelIndexSize.idx << ")" << std::endl;
    }

  // auto S = B2.getStorage();
  // std::vector<int> denseDim = {B2.getDimensions()[0]};
  // taco::storage::TensorStorage::LevelIndex d0Index(taco::util::copyToArray(denseDim), nullptr);
  // taco::storage::TensorStorage::LevelIndex d1Index(rowPtr, colIdx);
  // S.content->index[0] = d0Index;
  // S.content->index[1] = d1Index;
  // S.content->values = vals;
  std::cout << "after " << vals << ", " << rowPtr << ", " << colIdx << std::endl;
  //   S.setLevelIndex(0, d0Index);
  // S.setLevelIndex(1, d1Index);
  // S.setValues(vals);
    // free(vals);
    // free(rowPtr);
    // free(colIdx);
    // TACO_TIME_REPEAT(B2.pack(), 1, tr3)
    // std::cout << "pack" << std::endl;

    // Form a tensor-vector multiplication expression
    // taco::Var i, j(taco::Var::Sum);
    // a(i) = B(i,j) * c(j);

    // result_tensor = a;



    // // Compile the expression
    // std::cout << "compiling" << std::endl;
    // TACO_TIME_REPEAT(result_tensor.compile(), 1, tr5)
    // // std::cout << "compile took " << tr3 << std::endl;

    // // Assemble A's indices and numerically compute the result
    // std::cout << "assembling" << std::endl;
    // TACO_TIME_REPEAT(result_tensor.assemble(), 1, tr6)
    // // std::cout << "assemble took " << tr4 << std::endl;

    // std::cout << "computing" << std::endl;
    // TACO_TIME_REPEAT(result_tensor.compute(), 1, tr7)
    // // std::cout << "compute took " << tr5 << std::endl;

    // std::cout << "assigning output" << std::endl;
    // double* output_vals = result_tensor.getStorage().getValues();
    // size_t nvals = a_indices_->dim_size(0);
    // for (int i = 0; i < nvals; i++) {
    //   output(i) = output_vals[i];
    // }

    // std::cout << "assigning times" << std::endl;
    // int counter = 0;
    // for (auto tr : {tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7}) {
    //     // std::cout << tr.mean << ", ";
    //     time_output(counter++) = tr.mean;
    // }
    std::cout << "done with everything!" << std::endl;
  }
};

REGISTER_KERNEL_BUILDER(Name("RobLoader").Device(DEVICE_CPU), RobLoaderOp);
