import tensorflow as tf
zero_out_module = tf.load_op_library('./hello.so')
with tf.Session(''):
  print zero_out_module.zero_out([[17, 2], [3, 4]]).eval()

# Prints
#print tf.array([[1, 0], [0, 0]], dtype=int32)
