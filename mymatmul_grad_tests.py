"""
Tests for the inner product Tensorflow operation.

.. moduleauthor:: David Stutz
"""

import unittest
import numpy as np
import tensorflow as tf
import _mymatmul_grad
inner_product_module = tf.load_op_library('/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/mymatmul.so')

class InnerProductOpTest(unittest.TestCase):
    def test_raisesExceptionWithIncompatibleDimensions(self):
        print "test_raisesExceptionWithIncompatibleDimensions"
        with tf.Session(''):
            with self.assertRaises(ValueError):
                inner_product_module.my_matmul([[1, 2], [3, 4]], [1, 2]).eval()
            with self.assertRaises(ValueError):
                self.assertRaises(inner_product_module.my_matmul([1, 2, 3, 4], [1, 2]).eval(), ValueError)
            with self.assertRaises(ValueError):
                self.assertRaises(inner_product_module.my_matmul([[1, 2], [3, 4]], [1, 2, 3]).eval(), ValueError)
            
    def test_innerProductHardCoded(self):
        with tf.Session(''):
            print "test_innerProductHardCoded"
            result = tf.matmul([[1, 2], [3, 4]], [[1], [2]]).eval()
            # print result
            # result2 = tf.matmul([[1, 2], [3, 4]], [[1, 2]]).eval()
            # print result2
            # result3 = inner_product_module.my_matmul([[1, 2], [3, 4]], [[1, 2]]).eval()
            # print result3
            # result4 = inner_product_module.my_matmul([[1, 2], [3, 4]], [[1], [2]]).eval()
            # print result4
            self.assertEqual(result.shape[0], 2)
            self.assertEqual(result[0], 5)
            self.assertEqual(result[1], 11)

            # self.assertEqual(result2.shape[0], 2)
            # self.assertEqual(result2[0], 5)
            # self.assertEqual(result2[1], 11)
    
    def test_innerProductGradientXHardCoded(self):
        with tf.Session('') as sess:
            print "test_innerProductGradientXHardCoded"
            x = tf.placeholder(tf.float64, shape = (2))
            W = tf.constant(np.asarray([[1, 2], [3, 4]]).astype(np.float64))
            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_inner_product = inner_product_module.my_matmul(W, tf.reshape(x, [2, 1]))
            
            grad_x_tf = tf.gradients(Wx_tf, x)
            grad_x_inner_product = tf.gradients(Wx_inner_product, x)
            
            gradient_tf = sess.run(grad_x_tf, feed_dict = {x: np.asarray([1, 2]).astype(np.float64)})
            gradient_inner_product = sess.run(grad_x_inner_product, feed_dict = {x: np.asarray([1, 2]).astype(np.float64)})
            
            self.assertEqual(gradient_tf[0][0], gradient_inner_product[0][0])
            self.assertEqual(gradient_tf[0][1], gradient_inner_product[0][1])
    
    def test_innerProductGradientWHardCoded(self):

        with tf.Session('') as sess:
            print "test_innerProductGradientWHardCoded"
            x = tf.constant(np.asarray([1, 2]).astype(np.float64))
            W = tf.placeholder(tf.float64, shape = (2, 2))
            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_inner_product = inner_product_module.my_matmul(W, tf.reshape(x, [-1, 1]))
            
            grad_W_tf = tf.gradients(Wx_tf, W)
            grad_W_inner_product = tf.gradients(Wx_inner_product, W)
            
            gradient_tf = sess.run(grad_W_tf, feed_dict = {W: np.asarray([[1, 2], [3, 4]]).astype(np.float64)})
            gradient_inner_product = sess.run(grad_W_inner_product, feed_dict = {W: np.asarray([[1, 2], [3, 4]]).astype(np.float64)})
            
            self.assertEqual(gradient_tf[0][0][0], gradient_inner_product[0][0][0])
            self.assertEqual(gradient_tf[0][0][1], gradient_inner_product[0][0][1])
            self.assertEqual(gradient_tf[0][1][0], gradient_inner_product[0][1][0])
            self.assertEqual(gradient_tf[0][1][1], gradient_inner_product[0][1][1])
    
    def test_innerProductRandom(self):
        with tf.Session(''):
            print "test_innerProductRandom"
            n = 4
            m = 5
            
            for i in range(100):
                x_rand = np.random.randint(10, size = (n, 1))
                W_rand = np.random.randint(10, size = (m, n))
                result_rand = np.dot(W_rand, x_rand).astype(np.float64)
                # print "\nx_rand: ", x_rand, "\nW_rand: ", W_rand
                
                result = inner_product_module.my_matmul(W_rand, x_rand).eval()
                # print "\nresult: ", result, "\nresult_rand: ", result_rand
                np.testing.assert_array_equal(result, result_rand)
    
    def test_innerProductGradientXRandom(self):
        with tf.Session('') as sess:
            print "test_innerProductGradientXRandom"
            n = 4
            m = 5
            
            x = tf.placeholder(tf.float64, shape = (n))
            W = tf.placeholder(tf.float64, shape = (m, n))
            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_inner_product = inner_product_module.my_matmul(W, tf.reshape(x, [-1, 1]))
            
            grad_x_tf = tf.gradients(Wx_tf, x)
            grad_x_inner_product = tf.gradients(Wx_inner_product, x)
            
            for i in range(100):
                x_rand = np.random.randint(10, size = (n))
                W_rand = np.random.randint(10, size = (m, n))
                
                gradient_tf = sess.run(grad_x_tf, feed_dict = {x: x_rand, W: W_rand})
                gradient_inner_product = sess.run(grad_x_inner_product, feed_dict = {x: x_rand, W: W_rand})
                
                np.testing.assert_array_equal(gradient_tf, gradient_inner_product)
                
    def test_innerProductGradientWRandom(self):
        with tf.Session('') as sess:
            print "test_innerProductGradientWRandom"
            n = 4
            m = 5
            
            x = tf.placeholder(tf.float64, shape = (n))
            W = tf.placeholder(tf.float64, shape = (m, n))
            # print "\nx_tf: ", x, "\nW_tf: ", W
            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            # Wx_tf2 = tf.Print(Wx_tf, [Wx_tf], "tf Wx_tf")
            Wx_inner_product = inner_product_module.my_matmul(W, tf.reshape(x, [-1, 1]))
            # Wx_inner_product2 = tf.Print(Wx_inner_product, [Wx_inner_product], "tf Wx_inner_product")

            grad_W_tf = tf.gradients(Wx_tf, W)
            grad_W_inner_product = tf.gradients(Wx_inner_product, W)
            
            for i in range(100):
                x_rand = np.random.randint(10, size = (n))
                W_rand = np.random.randint(10, size = (m, n))
                # print "\nx_rand: ", x_rand, "\nW_rand: ", W_rand
                
                gradient_tf = sess.run(grad_W_tf, feed_dict = {x: x_rand, W: W_rand})
                gradient_inner_product = sess.run(grad_W_inner_product, feed_dict = {x: x_rand, W: W_rand})
                # print "\ngradient_tf: ", gradient_tf, "\ngradient_inner_product: ", gradient_inner_product

                W1 = sess.run(Wx_tf, feed_dict = {x: x_rand, W: W_rand})
                W2 = sess.run(Wx_inner_product, feed_dict = {x: x_rand, W: W_rand})
                # print "\nW1: ", W1, "\nW2: ", W2
                np.testing.assert_array_equal(W1, W2)
                
                np.testing.assert_array_equal(gradient_tf, gradient_inner_product)
                  
                
if __name__ == '__main__':
    unittest.main()
