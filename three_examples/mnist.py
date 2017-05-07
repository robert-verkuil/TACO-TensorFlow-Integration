# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import time
import _mymatmul_grad

FLAGS = None
mymatmul_module = tf.load_op_library('/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/mymatmul.so')

def main(use_mymatmul):
  # Import data
  mnist = input_data.read_data_sets("/home/ubuntu/tensorflow/tensorflow/core/user_ops/three_examples/Mnist_data/", one_hot=True)

  t0 = time.clock()
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  x = tf.Print(x, [tf.shape(x)], "x")
  W = tf.Variable(tf.zeros([784, 10]))
  W = tf.Print(W, [tf.shape(W)], "W")
  b = tf.Variable(tf.zeros([10]))
  b = tf.Print(b, [tf.shape(b)], "b")
  Wt = tf.transpose(W) # 10 x 784
  Wt = tf.Print(Wt, [tf.shape(Wt)], "Wt")
  print(Wt.get_shape())
  xt = tf.transpose(x) # 784 x None
  xt = tf.Print(xt, [tf.shape(xt)], "xt")
  print(xt.get_shape())
  if use_mymatmul:
    print("Using MyMatmul\n")
    xt = tf.cast(xt, tf.float64)
    Wt = tf.cast(Wt, tf.float64)
    b = tf.cast(b, tf.float64)
    
    prodt = mymatmul_module.my_matmul(Wt, xt) # 10 x None
    prodt = tf.Print(prodt, [tf.shape(prodt)], "prodt")
    y = tf.transpose(prodt) + b 
    y = tf.Print(y, [tf.shape(y)], "y")
    print("mymatmul y: " + str(y.get_shape()))
    # y = tf.matmul(x, W) + b
  else:
    print("NOT Using MyMatmul\n")
    prodt = tf.matmul(Wt, xt) # 10 x None
    y = tf.transpose(prodt) + b
    y = tf.Print(y, [tf.shape(y)], "y")
    print("tf y: " + str(y.get_shape()))

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])
  y_ = tf.Print(y_, [tf.shape(y_)], "y_")
  print("y_: " + str(y_.get_shape()))

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train

  t1 = time.clock()
  print("between start and training: {}".format(t1 - t0))
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    print("batch_xs.shape: ", batch_xs.shape, "batch_ys.shape: ", batch_ys.shape)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  t2 = time.clock()
  print("Total Train Time:           {}".format(t2 - t1))
  print("One Train Iteration:        {}".format((t2 - t1)/1000))

  # Test trained model
  t3 = time.clock()
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  result = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels})

  t4 = time.clock()
  print("Inference Time:             {}".format(t4 - t3))

  print("\nTotal Elapsed Time:         {}".format(t4 - t0))
  print("Final Accuracy:             {}\n\n".format(result))

if __name__ == '__main__':
  # parser = argparse.ArgumentParser()
  # parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
  #                     help='Directory for storing input data')
  # FLAGS, unparsed = parser.parse_known_args()
  # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  main(True)
  main(False)
