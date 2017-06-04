import tensorflow as tf
import numpy as np
import sys


mymatmul_module = tf.load_op_library('/home/ubuntu/tensorflow_with_debug/bazel-bin/tensorflow/core/user_ops/mymatmul.so')
varmatmul_module = tf.load_op_library('/home/ubuntu/tensorflow_with_debug/bazel-bin/tensorflow/core/user_ops/varmatmul.so')
noop_module = tf.load_op_library('/home/ubuntu/tensorflow_with_debug/bazel-bin/tensorflow/core/user_ops/noop.so')
loader_module = tf.load_op_library('/home/ubuntu/tensorflow_with_debug/bazel-bin/tensorflow/core/user_ops/loader.so')


def generate_sparse_m_and_dense_v(m, n, sparsity, sparse_pattern):
    W_rand = []
    x_rand = []
    if sparse_pattern == "ratio_based":
        # Generate the sparse matrix.
        W_rand = np.random.randint(0, high=100, size = (m, n))
        W_rand[W_rand > (sparsity*100)] = 0
        W_rand = W_rand.astype(np.float64)

        # Generate the vector.
        x_rand = np.random.randint(1, high=10, size = (n, 1))
        x_rand = x_rand.astype(np.float64)

    elif sparse_pattern == "n_per_line":
        # Generate the sparse matrix.
        rows = []
        for _ in range(m):
            row = np.random.randint(0, high=100, size = (1, n))
            row[row > (sparsity*100)] = 0
            rows.append(row)
        W_rand = np.vstack(rows)
        W_rand = W_rand.astype(np.float64)

        # Generate the vector.
        x_rand = np.random.randint(1, high=10, size = (n, 1))
        x_rand = x_rand.astype(np.float64)

    else:
        print("use a correct sparse pattern!")

    return W_rand, x_rand

def np_matmul_func(sparse_m, v):
    return np.dot(sparse_m, v)

# used to get a 
def tf_noop_func(sparse_m, v):
    with tf.Session(''):
        sz1 = tf.size(sparse_m).eval()
        # sz1 = tf.matmul(sparse_m, v).eval()
        return sz1

def my_noop_func(sparse_m, v):
    # for sp_fmt in sparsity_format:
    with tf.Session(''):
        result = noop_module.noop(sparse_m, v).eval()
        # result = mymatmul_module.my_matmul(sparse_m, v).eval()
        return result

# used to get a 
def tf_noop_func_feed(sparse_m_, v_):
    sess = tf.Session('')
    with sess:
        sparse_m = tf.placeholder(tf.double)
        v = tf.placeholder(tf.double)

        ret = tf.size(sparse_m)
        result = sess.run(ret, feed_dict={sparse_m: sparse_m_, v:v_})
        return result

def my_noop_func_feed(sparse_m_, v_):
    sess = tf.Session('')
    with sess:
        sparse_m = tf.placeholder(tf.double)
        v = tf.placeholder(tf.double)

        ret = noop_module.noop(sparse_m, v)
        result = sess.run(ret, feed_dict={sparse_m: sparse_m_, v:v_})
        return result


def my_matmul_func(sparse_m, v):
    # for sp_fmt in sparsity_format:
    with tf.Session(''):
        return mymatmul_module.my_matmul(sparse_m, v).eval()


def tf_matmul_func(sparse_m, v):
    with tf.Session(''):
        # config=tf.ConfigProto(intra_op_parallelism_threads=1)):
        return tf.matmul(sparse_m, v).eval()

if __name__ == '__main__':
	sparse_m, v = generate_sparse_m_and_dense_v(10000, 10000, 0.05, "ratio_based")

	if len(sys.argv) != 2:
		print("Invalid input, options = [np, tf_matmul, my_matmul, tf_noop, my_noop]")

	if sys.argv[1] == "np":
		np_matmul_func(sparse_m, v)
	elif sys.argv[1] == "tf_matmul":
		tf_matmul_func(sparse_m, v)
	elif sys.argv[1] == "my_matmul":
		my_matmul_func(sparse_m, v)
	elif sys.argv[1] == "tf_noop":
		tf_noop_func(sparse_m, v)
	elif sys.argv[1] == "my_noop":
		my_noop_func(sparse_m, v)
	elif sys.argv[1] == "tf_noop_feed":
		tf_noop_func_feed(sparse_m, v)
	elif sys.argv[1] == "my_noop_feed":
		my_noop_func_feed(sparse_m, v)
	else:
		print("Invalid string, options = [np, tf_matmul, my_matmul, tf_noop, my_noop]")
