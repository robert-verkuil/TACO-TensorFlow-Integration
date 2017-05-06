import unittest
import numpy as np
import tensorflow as tf
from subprocess import call

# import _mymatmul_grad
inner_product_module = tf.load_op_library('/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/mymatmul.so')

def tinytest():
    # call(["ldd", "/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/mymatmul.so"])
    with tf.Session(''):
        # call(["../../../../taco/build/bin/taco-tensor_times_vector"])
        print inner_product_module.my_matmul([[1, 1], [1, 1]], [[1], [1]]).eval()
        # print inner_product_module.my_matmul().eval()

if __name__ == '__main__':
    tinytest()