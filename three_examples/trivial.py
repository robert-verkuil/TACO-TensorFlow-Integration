import tensorflow as tf
import numpy as np
import timeit

mymatmul_module = tf.load_op_library('/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/mymatmul.so')

def trivial(no_tf, normal, mymatmul, check):
    n = 1500
    m = 1500
    
    for i in range(1):
        x_rand = np.random.randint(-50, high=2, size = (n, 1))
        W_rand = np.random.randint(-50, high=2, size = (m, n))
        x_rand[x_rand < 0] = 0
        W_rand[W_rand < 0] = 0

        if no_tf:
            result_rand = np.dot(W_rand, x_rand).astype(np.float64)
        if normal:
            with tf.Session(''):
                result = tf.matmul(W_rand.astype(np.float64), x_rand.astype(np.float64)).eval()
        if mymatmul:
            with tf.Session(''):
                result = mymatmul_module.my_matmul(W_rand, x_rand).eval()
        if check:
            np.testing.assert_array_equal(result, result_rand)
            

def no_tf():
    trivial(True, False, False, False)

def normal():
    trivial(False, True, False, False)

def mine():
    trivial(False, False, True, False)


print "no_tf: ", timeit.timeit(no_tf, number=20)
print "mine: ", timeit.timeit(mine, number=20)
print "normal: ", timeit.timeit(normal, number=20)
