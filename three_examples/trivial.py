from __future__ import print_function
import tensorflow as tf
import numpy as np
import timeit
import time

mymatmul_module = tf.load_op_library('/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/mymatmul.so')
varmatmul_module = tf.load_op_library('/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/varmatmul.so')

# print(*(0,1,2,3))

# # Function de-arger/anonymizer
# def anon(f, *arg):
#     def _anon():
#         f(*arg)
#     return _anon

# print "no_tf: ", timeit.timeit(
#     anon(trivial, False, tf.my_matmul),
#     number=20)

# print "mine: ", timeit.timeit(
#     anon(trivial, False, tf.my_matmul),
#     number=20)
# print "normal: ", timeit.timeit(
#     anon(trivial, False, mymatmul_module.my_matmul),
#     number=20)

    # .Attr("sparse_fmt: list(bool) = [true, true]")
    # .Attr("tns_file: string = \"\"")
    # .Attr("v_fmt_str: string = \"\"")


# def generate_sparse_m_and_dense_v_indices_and_values(m, n, sparsity, sparse_pattern):


def generate_sparse_m_and_dense_v(m, n, sparsity, sparse_pattern):
    W_rand = []
    x_rand = []
    if sparse_pattern == "ratio_based":
        # Generate the sparse matrix.
        W_rand = np.random.randint(0, high=100, size = (m, n))
        W_rand[W_rand > (1/sparsity)] = 0
        W_rand = W_rand.astype(np.float64)

        # Generate the vector.
        x_rand = np.random.randint(1, high=10, size = (n, 1))
        x_rand = x_rand.astype(np.float64)

    elif sparse_pattern == "n_per_line":
        # Generate the sparse matrix.
        rows = []
        for _ in range(m):
            row = np.random.randint(0, high=100, size = (1, n))
            row[row > (1/sparsity)] = 0
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


# used to get a 
def tf_noop_func(sparse_m, v):
    with tf.Session(''):
        sparse_m = tf.identity(sparse_m)
        # v = tf.identity(v)
        sparse_m.eval()
        # v.eval()

def tf_matmul_func(sparse_m, v):
    with tf.Session(''):
        return tf.matmul(sparse_m, v).eval()

def tf_sparse_matmul_func(sparse_m, v):
    # put stuff into a sparsetensor
    # print(sparse_m)
    indices = np.argwhere(sparse_m)
    # indices = indices[1:] # chop off the first element to make sure the correctness check breaks!
    values = sparse_m[zip(*indices)]
    # print(sparse_m[[[0,0], [0,1]]])
    # print(indices, values)
    # print(len(indices), len(values))
    sparse_t = tf.SparseTensor(indices=indices, values=values, dense_shape=sparse_m.shape)
    with tf.Session(''):
        result = tf.sparse_tensor_dense_matmul(sparse_t, v).eval()
        # print("result shape = ", result.shape)
        return result

def tf_matmul_with_sparsity_func(sparse_m, v):
    with tf.Session(''):
        return tf.matmul(sparse_m, v, a_is_sparse=True).eval()
    
def my_matmul_func(sparse_m, v):
    # for sp_fmt in sparsity_format:
    with tf.Session(''):
        return mymatmul_module.my_matmul(sparse_m, v).eval()

def varmatmul_func(sparse_m, v, sparse_fmt=[True, True]):
    # for sp_fmt in sparsity_format:
    with tf.Session(''):
        return varmatmul_module.var_matmul(sparse_m, v, sparse_fmt=sparse_fmt).eval()


functions = (
    # ref
    ("numpy_matmul_func", numpy_matmul_func),
    ("tf_noop_func", tf_noop_func),

    # tf stuff
    ("tf_matmul_func", tf_matmul_func),
    ("tf_sparse_matmul_func", tf_sparse_matmul_func),
    ("tf_matmul_with_sparsity_func", tf_matmul_with_sparsity_func),

    # my stuff
    ("my_matmul_func", my_matmul_func),
    ("varmatmul_func", varmatmul_func),
    )

sizes = (
    [30, 30],
    [50, 50],
    [100, 100],
        [500, 500],
        [1000,1000],
        [2000,2000],
    # [4000,4000],
    # [10000, 10000],
    )

# in the range [0,1]
sparsities = (
        0.01,
        0.03,
        0.05,
        0.1,
        0.15,
        0.2,
        0.3,
        # 0.4,
        0.5,
        0.75,
        1,

        1,
        0.75,
        0.5,
        0.3,
        0.2,
        0.15,
        0.1,
        0.05,
        0.03,
        0.01,
    )

# Which sparse format mymatmul will use
# True -> taco::Dense, False -> taco::Sparse
sparsity_format = (
    [True, True],
    [True, False],
    [False, True],
    [False, False],
    )

sparsity_pattern = (
    "ratio_based",
    "n_per_line",
    )

# # A quick tester to check for bugs with the very difficult stuff.
# # For use before running a huge suite of tests.
# def quick_test:
#     for f in functions:
#         for sparse_format in sparsity_pattern:
#             m, n = 10000, 10000


def run_one_test(fname, f, size, sparsity, sparse_pattern, check, sparse_fmt):
    m, n = size[0], size[1]

    # We split appart the print because f(sparse_m, v) has the side effect of printing some data
    # And we want the newline to come after that printed stuff
    print("{}, {}, {}, {}, {}, ".format(str(fname), str(m), str(n), str(sparsity), sparse_pattern), end='')
    start = time.time()
    sparse_m, v = generate_sparse_m_and_dense_v(m, n, sparsity, sparse_pattern)
    after_generation = time.time()
    print("{}, ".format(after_generation-start), end="")

    if sparse_fmt:
        result = f(sparse_m, v, sparse_fmt)
    else:
        result = f(sparse_m, v)
    after_computation = time.time()

    # We might need padding for the stuff spit out by the function
    if f != varmatmul_func:
        print(", "*8, end='')

    print("{}, ".format(after_computation-after_generation), end="")

    print('', end='\n')

    if f != tf_noop_func and check:
        good_result = np.dot(sparse_m, v)
        np.testing.assert_array_equal(result, good_result)


# Runs all combinations of the above variables.
# Puts the result into a csv for manipulation and graph generation
# in jupyter on the local machine.
def test_all(check):
    print("function_type, m, n, sparsity, sparsity_pattern, generation_time, TF_setup, tacotensor_init, tacotensor_insert, B_pack, c_pack, t_compile, t_assemble, t_compute, computation_time")
    for fname, f in functions:
        for size in sizes:
            for sparsity in sparsities:
                for sparse_pattern in sparsity_pattern:
                    if f ==  varmatmul_func:
                        for sparse_fmt in sparsity_format:
                            run_one_test(fname, f, size, sparsity, sparse_pattern, check, sparse_fmt)
                    else:
                        run_one_test(fname, f, size, sparsity, sparse_pattern, check, None)


if __name__ == '__main__':
    test_all(True)







