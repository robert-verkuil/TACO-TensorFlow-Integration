from __future__ import print_function
import tensorflow as tf
import numpy as np
import timeit
import time
import sys
from subprocess import call

mymatmul_module = tf.load_op_library('/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/mymatmul.so')
varmatmul_module = tf.load_op_library('/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/varmatmul.so')
noop_module = tf.load_op_library('/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/noop.so')



#########################################################################
#                         The Generator!
#########################################################################


# Since sparsity pattern didn't seem to make much of a difference before,
# We will just be using total random here...
#
# @param dims is array of dimensions (sizes of each axis)
# @param sparsity is the fill rate, given as a float
#
# @ return the three lists needed to make a Tensorflow tensor
def generate_sparse_coordinates_and_dense_v(dims, sparsity):
    seen = dict()
    coords = []
    size = np.prod(dims)

    # Fill up the coords list, without repeats, 
    # until we have our desired sparsity
    while len(coords)/size < sparsity:
        coord = tuple([np.random.randint(0, dims[i]) for i in range(dims)])
        if coord not in seen:
            coords.append(coord)
            seen[coord] = True

    values = numpy.random.rand(len(coords))

    dense_shape = tuple(dims)

    return coordinates, values, dense_shape



#########################################################################
#                         The Functions!
#########################################################################


    # with tf.Session(
    #     config=tf.ConfigProto(intra_op_parallelism_threads=1)):


# used to get a 
def tf_noop_sz_func(coords, values, dense_shape):
    with tf.Session(''):
        return tf.size(coords).eval()

# used to get a 
def tf_noop_load_sparse_tensor_func(coords, values, dense_shape):
    with tf.Session(''):
        sparse_t = tf.SparseTensor(coords, values, dense_shape)
        return tf.size(sparse_t).eval()


def tf_sparse_matmul_func(coords, values, dense_shape):
    # put stuff into a sparsetensor
    sparse_t = tf.SparseTensor(indices=coords, values=values, dense_shape=dense_shape)
    with tf.Session(''):
        result = tf.sparse_tensor_dense_matmul(sparse_t, v).eval()
        return result

def varmatmul_sparse_func(coords, values, dense_shape, sparse_fmt=[True, True]):
    # for sp_fmt in sparsity_format:
    with tf.Session(''):
        result, times = sess.run(varmatmul_module.var_matmul(sparse_m, v, sparse_fmt=sparse_fmt))
        return result, times

# def my_noop_func(sparse_m, v, sparse_fmt):
#     # for sp_fmt in sparsity_format:
#     with tf.Session(''):
#         result = noop_module.noop(sparse_m, v).eval()
#         return result



#########################################################################
#                         The Options!
#########################################################################


functions = {
    # Category: Baseline
    "tf_noop_sz_func" : tf_noop_sz_func,
    "tf_noop_load_sparse_tensor_func" : tf_noop_load_sparse_tensor_func,

    # Category: tf stuff
    "tf_sparse_matmul_func" : tf_sparse_matmul_func,

    # Category: my stuff
    "varmatmul_sparse_func" : varmatmul_sparse_func,
    # "my_noop_func" : my_noop_func,
    }

sizes = (
    # [1,1],
    # [50, 50],
    # [100, 100],
    # [500, 500],
    # [1000,1000],
    [2000,2000],
    # [4000,4000],
    [10000, 10000],
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
        # 0.1,
        # 0.15,
        # 0.2,
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

gen_sparse_matrix = True


# Right now I'm relying on the secret invariant that only varmatmul_func
# uses sparse_fmts at all
def run_one_test(fname, m, n, sparsity, sparse_pattern, check, sparse_fmt_name, gen_sparse_matrix):
    # m, n = size[0], size[1]
    f = functions[fname]
    sparse_fmt = sparsity_format[sparse_fmt_name]

    # We split appart the print because f(sparse_m, v) has the side effect of printing some data
    # And we want the newline to come after that printed stuff
    print("{}, {}, {}, {}, {}, ".format(str(fname), str(m), str(n), str(sparsity), sparse_pattern), end='')
    # Handling for sparse comparisons
    start = time.time()
    coords, vals, dense_shape = generate_sparse_coordinates_and_dense_v([m,n])
    after_generation = time.time()
    print("{}, ".format(after_generation-start), end="")

    if sparse_fmt:
        result, times = f(coords, vals, dense_shape, sparse_fmt)
    else:
        result = f(coords, vals, dense_shape)
    after_computation = time.time()

    # We might need padding for the stuff spit out by the function
    if f != varmatmul_sparse_func:
        print(", "*8, end='')
    else:
        for t in times.flatten()[:8]:
            print("{}, ".format(t), end='')

    print("{}".format(after_computation-after_generation), end="\n")

    # if 'noop_func' not in fname and check:
    #     good_result = np.dot(sparse_m, v)
    #     np.testing.assert_array_equal(result, good_result)


def call_run_one_test(options):
    time.sleep(0.1)
    call(["python", "trivial_sparse.py", "test_one"] + [str(opt) for opt in options])


# Runs all combinations of the above variables.
# Puts the result into a csv for manipulation and graph generation
# in jupyter on the local machine.
def test_all(check, gen_sparse_matrix):
    global gen_sparse_matrix
    print("function_type, m, n, sparsity, sparsity_pattern, generation_time, TF_setup, tacotensor_init, tacotensor_insert, B_pack, c_pack, t_compile, t_assemble, t_compute, computation_time")
    for fname, f in functions.iteritems():
        for (m, n) in sizes:
            for sparsity in sparsities:
                for sparse_pattern in sparsity_pattern:
                    if f == varmatmul_func:
                        for sparse_fmt_name, _ in sparsity_format.iteritems():
                            if sparse_fmt_name != -1:
                                call_run_one_test([fname, m, n, sparsity, sparse_pattern, check, sparse_fmt_name, gen_sparse_matrix])
                            # run_one_test(fname, m, n, sparsity, sparse_pattern, check, sparse_fmt_name)
                    else:
                        call_run_one_test([fname, m, n, sparsity, sparse_pattern, check, -1, gen_sparse_matrix])
                        # run_one_test(fname, m, n, sparsity, sparse_pattern, check, -1)


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("Need arg1 = [\"test_all\", \"test_one\"].\n" +
              "If test_one, then you need to specify all the args." + 
              "(They are ommitted here to avoid having to always change two code areas...")
    if sys.argv[1] == "test_all":
        test_all(True, False)
    elif sys.argv[1] == "test_one":
        # print("about to call one test with args = ", *sys.argv[2:9])
        fname           = sys.argv[2]
        m               = int(sys.argv[3])
        n               = int(sys.argv[4])
        sparsity        = float(sys.argv[5])
        sparse_pattern  = sys.argv[6]
        check           = [True if sys.argv[7]=="True" else False]
        sparse_fmt      = int(sys.argv[8])
        gen_sparse_matrix = bool(sys.argv[9])
        run_one_test(fname, m, n, sparsity, sparse_pattern, check, sparse_fmt, gen_sparse_matrix)







