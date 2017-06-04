from __future__ import print_function
import tensorflow as tf
import numpy as np
import timeit
import time
import sys
from subprocess import call

mymatmul_module = tf.load_op_library('/home/ubuntu/tensorflow_with_debug/bazel-bin/tensorflow/core/user_ops/mymatmul.so')
varmatmul_module = tf.load_op_library('/home/ubuntu/tensorflow_with_debug/bazel-bin/tensorflow/core/user_ops/varmatmul.so')
noop_module = tf.load_op_library('/home/ubuntu/tensorflow_with_debug/bazel-bin/tensorflow/core/user_ops/noop.so')

# def generate_sparse_m_and_dense_v_indices_and_values(m, n, sparsity, sparse_pattern):


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


# The plan:
# generate_sparse_m_and_dense_v
# then using that thingy, run multiple tests:
# vary size and sparsity - grid search
# so generate a csv with m, n, sparsity, sparsity type, 
# then use varmatmul to spit out its values

# We can send that file back to local machine for plotting

# do the plotting later in jupyter
# do this for each of the sparsity thingies
# generate two side by side plots with each sparsity type

def numpy_matmul_func(sparse_m, v):
    return np.dot(sparse_m, v)

def tf_noop_func(sparse_m, v):
    with tf.Session(''):
        sz1 = tf.size(sparse_m).eval()
        # sz1 = tf.matmul(sparse_m, v).eval()
        return sz1

def tf_noop_func_feed(sparse_m_, v_):
    sess = tf.Session('')
    with sess:
        sparse_m = tf.placeholder(tf.double)
        v = tf.placeholder(tf.double)

        ret = tf.size(sparse_m)
        result = sess.run(ret, feed_dict={sparse_m: sparse_m_, v:v_})
        return result

# def tf_noop_func2(sparse_m, v):
#     with tf.Session(''):
#         return tf.matmul([[0],[0]], [[0]]).eval()

def tf_matmul_func(sparse_m, v):
    with tf.Session(''):
        # config=tf.ConfigProto(intra_op_parallelism_threads=1)):
        return tf.matmul(sparse_m, v).eval()

def tf_matmul_func_feed(sparse_m_, v_):
    sess = tf.Session('')
    with sess:
        sparse_m = tf.placeholder(tf.double)
        v = tf.placeholder(tf.double)

        ret = tf.matmul(sparse_m, v)
        result = sess.run(ret, feed_dict={sparse_m: sparse_m_, v:v_})
        return result

# def tf_matmul_func2(sparse_m, v):
#     NUM_THREADS = 16
#     with tf.Session(
#         config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)):
#         return tf.matmul(sparse_m, v).eval()

def tf_sparse_matmul_func(sparse_m, v):
    # put stuff into a sparsetensor
    start = time.time()
    indices = np.argwhere(sparse_m)
    values = sparse_m[zip(*indices)]
    time_to_ignore = time.time() - start

    sparse_t = tf.SparseTensor(indices=indices, values=values, dense_shape=sparse_m.shape)
    with tf.Session(''):
        result = tf.sparse_tensor_dense_matmul(sparse_t, v).eval()
        # print("result shape = ", result.shape)
        return result, time_to_ignore

# def tf_sparse_matmul_func_feed(sparse_m_, v_):
#     sess = tf.Session('')
#     with sess:
#         sparse_m = tf.placeholder(tf.double)
#         v = tf.placeholder(tf.double)

#         ret = tf.sparse_tensor_dense_matmul(sparse_m, v)
#         result = sess.run(ret, feed_dict={sparse_m: sparse_m_, v:v_})
#         return result

def tf_matmul_with_sparsity_func(sparse_m, v):
    with tf.Session(''):
        return tf.matmul(sparse_m, v, a_is_sparse=True).eval()

def tf_matmul_with_sparsity_func_feed(sparse_m_, v_):
    sess = tf.Session('')
    with sess:
        sparse_m = tf.placeholder(tf.double)
        v = tf.placeholder(tf.double)

        ret = tf.matmul(sparse_m, v, a_is_sparse=True)
        result = sess.run(ret, feed_dict={sparse_m: sparse_m_, v:v_})
        return result
    
def my_matmul_func(sparse_m, v):
    # for sp_fmt in sparsity_format:
    with tf.Session(''):
        return mymatmul_module.my_matmul(sparse_m, v).eval()

def my_matmul_func_feed(sparse_m_, v_):
    sess = tf.Session('')
    with sess:
        sparse_m = tf.placeholder(tf.double)
        v = tf.placeholder(tf.double)

        ret = mymatmul_module.my_matmul(sparse_m, v)
        result = sess.run(ret, feed_dict={sparse_m: sparse_m_, v:v_})
        return result

def varmatmul_func(sparse_m, v, sparse_fmt=[True, True]):
    # for sp_fmt in sparsity_format:
    sess = tf.Session('')
    with sess:
        result, times = sess.run(varmatmul_module.var_matmul(sparse_m, v, sparse_fmt=sparse_fmt))
        return result, times

def varmatmul_func_feed(sparse_m_, v_, sparse_fmt_=[True, True]):
    sess = tf.Session('')
    with sess:
        sparse_m = tf.placeholder(tf.double)
        v = tf.placeholder(tf.double)

        ret = varmatmul_module.var_matmul(sparse_m, v, sparse_fmt=sparse_fmt_)
        result, times = sess.run(ret, feed_dict={sparse_m: sparse_m_, v:v_})
        return result, times

def my_noop_func(sparse_m, v):
    # for sp_fmt in sparsity_format:
    with tf.Session(''):
        result = noop_module.noop(sparse_m, v).eval()
        # result = mymatmul_module.my_matmul(sparse_m, v).eval()
        return result

def my_noop_func_feed(sparse_m_, v_):
    sess = tf.Session('')
    with sess:
        sparse_m = tf.placeholder(tf.double)
        v = tf.placeholder(tf.double)

        ret = noop_module.noop(sparse_m, v)
        result = sess.run(ret, feed_dict={sparse_m: sparse_m_, v:v_})
        return result


functions = {
    # # Category: Baseline
    # "numpy_matmul_func" : numpy_matmul_func,
    # "tf_noop_func" : tf_noop_func,

    # # Category: tf stuff
    # "tf_matmul_func" : tf_matmul_func,
    # "tf_sparse_matmul_func_bare" : tf_sparse_matmul_func,
    # "tf_matmul_with_sparsity_func" : tf_matmul_with_sparsity_func,

    # # Category: my stuff
    # "my_matmul_func" : my_matmul_func,
    # "varmatmul_func" : varmatmul_func,
                                                    # "my_noop_func" : my_noop_func,


    # Category: Baseline
    "numpy_matmul_func" : numpy_matmul_func,
    "tf_noop_func_feed" : tf_noop_func_feed,

    # Category: tf stuff
    "tf_matmul_func_feed" : tf_matmul_func_feed,
    # "tf_sparse_matmul_func_bare_feed" : tf_sparse_matmul_func_bare_feed,
    "tf_matmul_with_sparsity_func_feed" : tf_matmul_with_sparsity_func_feed,

    # Category: my stuff
    "my_matmul_func_feed" : my_matmul_func_feed,
    "varmatmul_func_feed" : varmatmul_func_feed,
    }

sizes = (
    # [1,1],
    # [50, 50],
    # [100, 100],
    # [500, 500],
    # [1000,1000],
    [2000,2000],
    # [4000,4000],
    # [10000, 10000],
    )

# in the range [0,1]
sparsities = (
        # 0.00001,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.1,
        0.15,
        0.2,
        # 0.3,
        # 0.5,
        # 0.75,
        # 1,
    )

# Which sparse format mymatmul will use
# True -> taco::Dense, False -> taco::Sparse
sparsity_format = {
    -1 : None,
    0 : [True, True],
    1 : [True, False],
    2 : [False, True],
    3 : [False, False],
    }

sparsity_pattern = (
    "ratio_based",
    # "n_per_line",
    )



# Right now I'm relying on the secret invariant that only varmatmul_func
# uses sparse_fmts at all
def run_one_test(fname, m, n, sparsity, sparse_pattern, check, sparse_fmt_name):
    # m, n = size[0], size[1]
    f = functions[fname]
    sparse_fmt = sparsity_format[sparse_fmt_name]

    # We split appart the print because f(sparse_m, v) has the side effect of printing some data
    # And we want the newline to come after that printed stuff
    print("{}, {}, {}, {}, {}, ".format(str(fname), str(m), str(n), str(sparsity), sparse_pattern), end='')

    # Handling for dense comparisons
    start = time.time()
    sparse_m, v = generate_sparse_m_and_dense_v(m, n, sparsity, sparse_pattern)
    after_generation = time.time()
    print("{}, ".format(after_generation-start), end="")


    if "varmatmul" in fname:
        result, times = f(sparse_m, v, sparse_fmt)
    elif f == tf_sparse_matmul_func:
        result, time_to_ignore = f(sparse_m, v)
    else:
        result = f(sparse_m, v)

    after_computation = time.time()

    # We might need padding for the stuff spit out by the function
    if "varmatmul" not in fname:
        print(", "*8, end='')
    else:
        for t in times.flatten()[:8]:
            print("{}, ".format(t), end='')

    if f == tf_sparse_matmul_func:
        print("{}".format(after_computation-after_generation-time_to_ignore), end="")
    else:
        print("{}".format(after_computation-after_generation), end="")

    print('', end='\n')

    if 'noop' not in fname and check:
        good_result = np.dot(sparse_m, v)
        np.testing.assert_array_equal(result, good_result)


def call_run_one_test(options):
    time.sleep(0.1)
    call(["python", "trivial.py", "test_one"] + [str(opt) for opt in options])


# Runs all combinations of the above variables.
# Puts the result into a csv for manipulation and graph generation
# in jupyter on the local machine.
def test_all(check):
    print("function_type, m, n, sparsity, sparsity_pattern, generation_time, TF_setup, tacotensor_init, tacotensor_insert, B_pack, c_pack, t_compile, t_assemble, t_compute, computation_time")
    for fname, f in functions.iteritems():
        for (m, n) in sizes:
            for sparsity in sparsities:
                for sparse_pattern in sparsity_pattern:
                    if "varmatmul" in fname:
                        for sparse_fmt_name, _ in sparsity_format.iteritems():
                            if sparse_fmt_name != -1:
                                call_run_one_test([fname, m, n, sparsity, sparse_pattern, check, sparse_fmt_name])
                            # run_one_test(fname, m, n, sparsity, sparse_pattern, check, sparse_fmt_name)
                    else:
                        call_run_one_test([fname, m, n, sparsity, sparse_pattern, check, -1])
                        # run_one_test(fname, m, n, sparsity, sparse_pattern, check, -1)


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("Need arg1 = [\"test_all\", \"test_one\"].\n" +
              "If test_one, then you need to specify all the args." + 
              "(They are ommitted here to avoid having to always change two code areas...")
    if sys.argv[1] == "test_all":
        test_all(False)
    elif sys.argv[1] == "test_one":
        # print("about to call one test with args = ", *sys.argv[2:9])
        fname           = sys.argv[2]
        m               = int(sys.argv[3])
        n               = int(sys.argv[4])
        sparsity        = float(sys.argv[5])
        sparse_pattern  = sys.argv[6]
        check           = [True if sys.argv[7]=="True" else False]
        sparse_fmt      = int(sys.argv[8])
        run_one_test(fname, m, n, sparsity, sparse_pattern, check, sparse_fmt)







