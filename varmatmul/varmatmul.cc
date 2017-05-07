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

REGISTER_OP("VarMatmul")
    .Input("sparse_m: double")
    .Input("v: double")
    .Attr("sparse_fmt: list(bool) = [true, true]")
    .Attr("tns_file: string = \"\"")
    .Attr("v_fmt_str: string = \"\"")
    .Output("result: double")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .SetShapeFn(::tensorflow::shape_inference::MatMulShape);

class VarMatmulOp : public OpKernel {
 private:
    std::vector<bool> sparse_fmt_;
    std::string tns_file_;
    std::string v_fmt_str_;

 public:
  explicit VarMatmulOp(OpKernelConstruction* context) : OpKernel(context) {
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
    const Tensor& input_tensor_a = context->input(0);
    auto sparse_m = input_tensor_a.matrix<double>();
    const Tensor& input_tensor_b = context->input(1);
    auto v = input_tensor_b.flat<double>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    const TensorShape& weights_shape = input_tensor_a.shape();
    TensorShape output_shape;
    output_shape.AddDim(weights_shape.dim_size(0));
    output_shape.AddDim(1);
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
    auto output = output_tensor->flat<double>();
    timer0.stop();
    tr0 = timer0.getResult();



    ////////////////////////////// TACO STUFF //////////////////////////////////
    // taco::util::Timer timer;

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


    // std::cout << "tns_file_ =  " << tns_file_ << std::endl;
    if (tns_file_ == "") {
        taco::util::Timer timer1;
        timer1.start();
        taco::Tensor<double> a({(int)v.size()},   dv);
        taco::Tensor<double> B({(int)input_tensor_a.shape().dim_size(0),
                       (int)input_tensor_a.shape().dim_size(1)},
                       sp_mtx_fmt_to_use);
        taco::Tensor<double> c({(int)v.size()},   dv);
        timer1.stop();
        tr1 = timer1.getResult();
        // FOR SPMV USAGE
        // Populate the sparse matrix
        taco::util::Timer timer2;
        timer2.start();
        for (int r = 0; r < input_tensor_a.shape().dim_size(0); r++) {
          for (int c = 0; c < input_tensor_a.shape().dim_size(1); c++) {
            if (sparse_m(r, c) != 0) {
              B.insert({r, c}, sparse_m(r, c));
            }
          }
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
        // B.pack();
        TACO_TIME_REPEAT(c.pack(), 1, tr4)
        // c.pack();

        // Form a tensor-vector multiplication expression
        taco::Var i, j(taco::Var::Sum);
        a(i) = B(i,j) * c(j);

        result_tensor = a;
    } else {
        // FOR TNS FILE USAGE
        // std::cout << "Trying to load tns_file_ =  " << tns_file_ << std::endl;
        taco::util::Timer timer1;
        timer1.start();
        taco::TensorBase T = taco::io::tns::readTensor(tns_file_, "Tensor");
        // std::cout << "FINISHED loading the tns_file_ =  " << tns_file_ << std::endl;
        timer1.stop();
        tr1 = timer1.getResult();
        T.setFormat(sp_mtx_fmt_to_use);

        // Make the Dense Vector
        int vector_length = T.getDimensions()[T.getOrder() - 1];
        // std::cout << "vector length = " << vector_length << std::endl;
        taco::Tensor<double> c({vector_length},   dv);
        taco::util::FillMethod fM = strToFill(v_fmt_str_);
        taco::util::fillTensor(c, fM);

        // Make Iteration Variables
        // std::cout << "making a_vars" << std::endl;
        std::vector<taco::Var> a_vars;
        for (int i = 0; i < T.getOrder() - 1; i++) {
            a_vars.push_back(taco::Var());
        }

        // std::cout << "making b_vars" << std::endl;
        std::vector<taco::Var> B_Vars;
        B_Vars.insert(B_Vars.end(), a_vars.begin(), a_vars.end());
        B_Vars.push_back(taco::Var(taco::Var::Sum));

        // std::cout << "making c_vars" << std::endl;
        std::vector<taco::Var> c_Vars;
        c_Vars.push_back(B_Vars[B_Vars.size()-1]);

        // Get Dimensions of A from dimensions of T
        std::vector<int> T_dims = T.getDimensions();
        std::vector<int> A_dims;
        A_dims.insert(A_dims.end(), T_dims.begin(), T_dims.end()-1);
        // std::cout << "A_dims has length = " << A_dims.size() << "T_dims has length = " << T_dims.size() << std::endl;

        // Get Format of A from format of T
        std::vector<taco::LevelType> A_fmts_list;
        A_fmts_list.insert(A_fmts_list.end(), fmts_list.begin(), fmts_list.end()-1);
        // std::cout << "A_fmts_list is " << A_fmts_list.size() << " long" << std::endl;
        taco::Format result_fmt_to_use(A_fmts_list);

        // Create the result tensor
        taco::Tensor<double> A(A_dims, result_fmt_to_use);

        // Pack data as described by the formats
        // std::cout << "packing c" << std::endl;
        TACO_TIME_REPEAT(T.pack(), 1, tr3)
        // std::cout << "packing T took " << tr1 << std::endl;

        // std::cout << "packing c" << std::endl;
        TACO_TIME_REPEAT(c.pack(), 1, tr4)
        // std::cout << "packing c took " << tr2 << std::endl;

        // Use the final expression
        // std::cout << "setting up expression" << std::endl;
        A(a_vars) = taco::Read(T,B_Vars) * c(c_Vars); 

        result_tensor = A;

        // // print out the fmts
        // for (auto ds : {a_vars, B_Vars, c_Vars}) {
        //     std::cout << "(" << taco::util::join(ds) << "), ";
        // }

        // // print out the dimenions
        // for (auto ds : {A.getDimensions(), T.getDimensions(), c.getDimensions()}) {
        //     std::cout << "(" << taco::util::join(ds) << "), ";
        // }
    }

    // Compile the expression
    // std::cout << "compiling" << std::endl;
    TACO_TIME_REPEAT(result_tensor.compile(), 1, tr5)
    // std::cout << "compile took " << tr3 << std::endl;

    // Assemble A's indices and numerically compute the result
    // std::cout << "assembling" << std::endl;
    TACO_TIME_REPEAT(result_tensor.assemble(), 1, tr6)
    // std::cout << "assemble took " << tr4 << std::endl;

    // std::cout << "computing" << std::endl;
    TACO_TIME_REPEAT(result_tensor.compute(), 1, tr7)
    // std::cout << "compute took " << tr5 << std::endl;

    double* output_vals = result_tensor.getStorage().getValues();
    size_t nvals = weights_shape.dim_size(0);
    for (int i = 0; i < nvals; i++) {
      output(i) = output_vals[i];
    }

    for (auto tr : {tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7}) {
        std::cout << tr.mean << ", ";
    }
    // vector<double> times = {tr0.mean, tr1.mean, tr2.mean, tr3.mean, tr4.mean, tr5.mean, tr6.mean, tr7.mean};
    // std::cout << taco::util::join(times);
    // std::cout << std::endl;
    // std::cout << tr0.mean << ", " << tr1.mean << ", " << tr2.mean << ", " << tr3.mean << ", " << tr4.mean << ", " << tr5.mean  << std::endl;
  }
};

REGISTER_KERNEL_BUILDER(Name("VarMatmul").Device(DEVICE_CPU), VarMatmulOp);
