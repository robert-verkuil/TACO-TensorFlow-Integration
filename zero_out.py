import tensorflow as tf
zero_out_module = tf.load_op_library('/home/ubuntu/tensorflow/tensorflow/core/user_ops/zero_out.so')
with tf.Session(''):
  print zero_out_module.zero_out([[17, 2], [3, 4]]).eval()

# Prints
#print tf.array([[1, 0], [0, 0]], dtype=int32)
