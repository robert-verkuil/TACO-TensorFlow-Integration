load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

cc_library(
    name = "taco",
    hdrs = glob(["include/**/*.h"]),
    srcs = ["libtaco.a"],
    visibility = ["//visibility:public"],

)

tf_custom_op_library(
    name = "fact.so",
    srcs = ["fact.cc"],
    deps = [
    ],
)

tf_custom_op_library(
    name = "noop.so",
    srcs = ["noop.cc"],
    deps = [
    ],
)

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
    deps = [
        ":taco"
    ],
)

tf_custom_op_library(
    name = "mymatmul.so",
    srcs = ["mymatmul.cc"],
    deps = [
        ":taco"
    ],
)

tf_custom_op_library(
    name = "mymatmul_grad.so",
    srcs = ["mymatmul_grad.cc"],
    deps = [
        ":taco"
    ],
)

tf_custom_op_library(
    name = "varmatmul.so",
    srcs = ["varmatmul/varmatmul.cc"],
    deps = [
        ":taco"
    ],
)

tf_custom_op_library(
    name = "varmatmul_sparse.so",
    srcs = ["varmatmul/varmatmul_sparse.cc"],
    deps = [
        ":taco"
    ],
)

tf_custom_op_library(
    name = "loader.so",
    srcs = ["varmatmul/loader.cc"],
    deps = [
        ":taco"
    ],
)

tf_custom_op_library(
    name = "spmv_taco_input.so",
    srcs = ["varmatmul/spmv_taco_input.cc"],
    deps = [
        ":taco"
    ],
)

