import tensorflow as tf
import numpy as np

mymatmul_module = tf.load_op_library('/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/mymatmul.so')
with tf.Session(''):
  sparse_m = np.array([[17., 2.], [3., 4.]]).astype(np.float64)
  v = np.array([[5.], [7.]]).astype(np.float64)
  print mymatmul_module.my_matmul(sparse_m, v).eval()

# Prints
#print tf.array([[1, 0], [0, 0]], dtype=int32)
