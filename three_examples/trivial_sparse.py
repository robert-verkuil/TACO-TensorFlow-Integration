from __future__ import print_function
import tensorflow as tf
import numpy as np
import timeit
import time
import sys
from subprocess import call
from tensorflow.python.client import timeline

mymatmul_module = tf.load_op_library('/home/ubuntu/tensorflow_with_debug/bazel-bin/tensorflow/core/user_ops/mymatmul.so')
varmatmul_module = tf.load_op_library('/home/ubuntu/tensorflow_with_debug/bazel-bin/tensorflow/core/user_ops/varmatmul.so')
varmatmul_sparse_module = tf.load_op_library('/home/ubuntu/tensorflow_with_debug/bazel-bin/tensorflow/core/user_ops/varmatmul_sparse.so')
noop_module = tf.load_op_library('/home/ubuntu/tensorflow_with_debug/bazel-bin/tensorflow/core/user_ops/noop.so')

loader_module = tf.load_op_library('/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/loader.so')
spmv_taco_input_module = tf.load_op_library('/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/spmv_taco_input.so')



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
    while float(len(coords))/float(size) < sparsity:
        # print(float(len(coords))/float(size), sparsity)
        coord = tuple([np.random.randint(0, dims[i]) for i in range(len(dims))])
        if coord not in seen:
            coords.append(coord)
            seen[coord] = True

    values = np.random.rand(len(coords))

    dense_shape = tuple(dims)

    v = np.random.random_sample((dims[-1],))

    return coords, values, dense_shape, np.array([v]).T



#########################################################################
#                         The Functions!
#########################################################################


    # with tf.Session(
    #     config=tf.ConfigProto(intra_op_parallelism_threads=1)):


# used to get a 
def tf_noop_sz_func(coords, values, dense_shape, v):
    with tf.Session(''):
        return tf.size(coords).eval()

# used to get a 
def tf_noop_load_sparse_tensor_func(coords, values, dense_shape, v):
    with tf.Session(''):
        sparse_t = tf.SparseTensor(coords, values, dense_shape)
        return tf.size(sparse_t).eval()

# used to get a 
def tf_noop_load_sparse_tensor_func2(coords_, values_, dense_shape_, v_):
    sess = tf.Session('')
    with sess:
        indices = tf.placeholder(tf.int64, [None, 2])
        values = tf.placeholder(tf.double, [None])
        dense_shape = tf.placeholder(tf.int64, [2,])
        v = tf.placeholder(tf.double, [None, 1])

        sparse_t = tf.SparseTensor(indices, values, dense_shape)
        ret = tf.size(sparse_t)
        result = sess.run(ret, feed_dict={indices: coords_, values: values_, dense_shape: dense_shape_, v:v_})
        return result


def tf_sparse_matmul_func(coords, values, dense_shape, v):
    # put stuff into a sparsetensor
    sparse_t = tf.SparseTensor(indices=coords, values=values, dense_shape=dense_shape)
    sess = tf.Session('')
    with sess:
        result = tf.sparse_tensor_dense_matmul(sparse_t, v).eval()
        return result

def tf_sparse_matmul_func2(coords, values, dense_shape, v):
    # put stuff into a sparsetensor
    sess = tf.Session('')
    with sess:
        sparse_t = tf.SparseTensor(indices=coords, values=values, dense_shape=dense_shape)
        result = tf.sparse_tensor_dense_matmul(sparse_t, v).eval()
        return result

def tf_sparse_matmul_func3(coords, values, dense_shape, v):
    # put stuff into a sparsetensor
    sess = tf.Session('')
    with sess:
        sparse_t = tf.SparseTensor(indices=coords, values=values, dense_shape=dense_shape)
        result = sess.run(tf.sparse_tensor_dense_matmul(sparse_t, v))
        return result

def tf_sparse_matmul_func4(coords_, values_, dense_shape_, v_):

    # print("memory usage = {}".format(len(coords_)*2*4+len(values_)*1*8+len(v_)*8))

    # put stuff into a sparsetensor
    sess = tf.Session('')
    with sess:
        indices = tf.placeholder(tf.int64, [None, 2])
        values = tf.placeholder(tf.double, [None])
        dense_shape = tf.placeholder(tf.int64, [2,])
        v = tf.placeholder(tf.double, [None, 1])

        sparse_t = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
        ret = tf.sparse_tensor_dense_matmul(sparse_t, v)
        result = sess.run(ret, feed_dict={indices: coords_, values: values_, dense_shape: dense_shape_, v:v_})
        return result

def tf_sparse_matmul_func5(coords_, values_, dense_shape_, v_):

    # print("memory usage = {}".format(len(coords_)*2*4+len(values_)*1*8+len(v_)*8))

    # put stuff into a sparsetensor
    start = time.time()
    sess = tf.Session('')
    with sess:
        indices = tf.placeholder(tf.int64, [None, 2])
        values = tf.placeholder(tf.double, [None])
        dense_shape = tf.placeholder(tf.int64, [2,])
        v = tf.placeholder(tf.double, [None, 1])

        # for i in range(3):
        sparse_t_ = sess.run(tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape), 
                        feed_dict={indices: coords_, values: values_, dense_shape: dense_shape_, v:v_})
    
        mid = time.time()
            # print("loading iteration {} took {}".format(i, mid-start))
            # start = mid


        sparse_t = tf.sparse_placeholder(tf.float64)

        # for i in range(3):
        result = sess.run(tf.sparse_tensor_dense_matmul(sparse_t, v), 
                        feed_dict={sparse_t: sparse_t_, v: v_})
        end = time.time()
            # print("loading iteration {} took {}".format(i, end-mid))
            # mid = end
        return result, mid

def varmatmul_sparse_func(coords, values, dense_shape, v, sparse_fmt=[True, False]):
    # for sp_fmt in sparsity_format:
    sess = tf.Session('')
    with sess:
        result, times = sess.run(varmatmul_sparse_module.var_matmul_sparse(coords, values, dense_shape, v, sparse_fmt=sparse_fmt))
        return result, times

def varmatmul_sparse_func2(coords_, values_, dense_shape_, v_, sparse_fmt_=[True, False]):
    # put stuff into a sparsetensor

    # for sp_fmt in sparsity_format:
    sess = tf.Session('')
    with sess:
        indices = tf.placeholder(tf.int64, [None, 2])
        values = tf.placeholder(tf.double, [None])
        dense_shape = tf.placeholder(tf.int64, [2,])
        v = tf.placeholder(tf.double, [None, 1])
        # sparse_fmt = tf.placeholder(tf.double, [1, 2])



        # sparse_t = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
        # print("in function: ", sparse_fmt_)
        ret = varmatmul_sparse_module.var_matmul_sparse(indices, values, dense_shape, v, sparse_fmt=[True, False])
        result, times = sess.run(ret, feed_dict={indices: coords_, values: values_, dense_shape: dense_shape_, v:v_})

        return result, times, None


# def split_func2_old(coords_, values_, dense_shape_, v_, sparse_fmt_=[True, False]):
#     # put stuff into a sparsetensor

#     # for sp_fmt in sparsity_format:
#     sess = tf.Session('')
#     start = time.time()
#     with sess:
#         run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#         run_metadata = tf.RunMetadata()

#         indices = tf.placeholder(tf.int64, [None, 2])
#         values = tf.placeholder(tf.double, [None])
#         dense_shape = tf.placeholder(tf.int64, [2,])
#         v = tf.placeholder(tf.double, [None, 1])
#         # sparse_fmt = tf.placeholder(tf.double, [1, 2])

#         # sparse_t = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
#         # print("in function: ", sparse_fmt_)
#         o1, o2, o3, o4, t = loader_module.rob_loader(indices, values, dense_shape, v, sparse_fmt=[True, False])
#         # print("o1 ", o1, ",o2 ", o2, " ,o3 ", o3, " o4, ", o4, ", t ",t)
#         # o1_ = tf.Print(o1, [t], '', summarize=10)
#         r, t2 = spmv_taco_input_module.spmv_taco_input(o1, o2, o3, o4, v, sparse_fmt=[True, False])
#         mid = time.time()
#         result, times, times2 = sess.run([r, t, t2], 
#                                         feed_dict={indices: coords_, values: values_, dense_shape: dense_shape_, v:v_},
#                                         options=run_options, run_metadata=run_metadata)
#         # print("loading times: ", times.flatten()[:8], "spmv_taco_input_processor: ", times2.flatten()[:8])
#         all_times = list(times.flatten()[:8]) + list(times2.flatten()[:8])
#         # print(all_times, len(all_times))

#         end = time.time()
#         print("graph construction took {} milliseconds to run".format(mid-start))
#         print("split_func2 took {} milliseconds to run".format(end-start))


#         # Create the Timeline object, and write it to a json
#         tl = timeline.Timeline(run_metadata.step_stats)
#         ctf = tl.generate_chrome_trace_format()
#         with open('timeline.json', 'w') as f:
#             f.write(ctf)


#         return result, all_times

def split_func2(coords_, values_, dense_shape_, v_, sparse_fmt_=[True, False]):
    # put stuff into a sparsetensor

    # for sp_fmt in sparsity_format:
    sess = tf.Session('')
    start = time.time()
    with sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        indices = tf.placeholder(tf.int64, [None, 2])
        values = tf.placeholder(tf.double, [None])
        dense_shape = tf.placeholder(tf.int64, [2,])
        v = tf.placeholder(tf.double, [None, 1])
        # sparse_fmt = tf.placeholder(tf.double, [1, 2])

        # sparse_t = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
        # print("in function: ", sparse_fmt_)

        # for i in range(5):
        o1_, o2_, o3_, o4_, t = sess.run(loader_module.rob_loader(indices, values, dense_shape, v, sparse_fmt=[True, False]),
                                        feed_dict={indices: coords_, values: values_, dense_shape: dense_shape_, v:v_})
        mid = time.time()
            # print("loading iteration {} took {}".format(i, mid-start))
            # start = mid

        memory_usage = len(o1_)*4+len(o2_)*4+len(o3_)*8+len(o4_)*4+len(v_)*8
        # print("memory usage = {}".format(memory_usage))

        o1 = tf.placeholder(tf.int32)
        o2 = tf.placeholder(tf.int32)
        o3 = tf.placeholder(tf.double)
        o4 = tf.placeholder(tf.int32)
        v = tf.placeholder(tf.double)
        ret = spmv_taco_input_module.spmv_taco_input(o1, o2, o3, o4, v, sparse_fmt=[True, False])

        # for i in range(5):
        r, t2 = sess.run(ret,
                            feed_dict={o1: o1_, o2: o2_, o3: o3_, o4: o4_, v:v_},
                            options=run_options, run_metadata=run_metadata)
        end = time.time()
            # print("using iteration {} took {}".format(i, end-mid))
            # mid = end

        # print("loading times: ", times.flatten()[:8], "spmv_taco_input_processor: ", times2.flatten()[:8])
        all_times = list(t.flatten()[:8]) + list(t2.flatten()[:8])
        # print(all_times, len(all_times))

        # print("graph construction took {} milliseconds to run".format(mid-start))
        # print("split_func2 took {} milliseconds to run".format(end-mid))


        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)


        return r, all_times, mid, memory_usage


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
    # "tf_noop_sz_func" : tf_noop_sz_func,
    # "tf_noop_load_sparse_tensor_func" : tf_noop_load_sparse_tensor_func,
    # "tf_noop_load_sparse_tensor_func2" : tf_noop_load_sparse_tensor_func2,

    # Category: tf stuff
    # "tf_sparse_matmul_func" : tf_sparse_matmul_func,
    # "tf_sparse_matmul_func2" : tf_sparse_matmul_func2,
    # "tf_sparse_matmul_func3" : tf_sparse_matmul_func3,
    # "tf_sparse_matmul_func4" : tf_sparse_matmul_func4,
    "tf_sparse_matmul_func5" : tf_sparse_matmul_func5,

    # Category: my stuff
    # "varmatmul_sparse_func" : varmatmul_sparse_func,
    # "varmatmul_sparse_func2" : varmatmul_sparse_func2,
    # "split_func2" : split_func2,
# "my_noop_func" : my_noop_func,

    # Category: split stuff

    }

sizes = (
    # [1,1],
    # [50, 50],
    # [100, 100],
    # [500, 500],
    # [1000,1000],
    # [2000,2000],
    # [4000,4000],
    [10000, 10000],
    # [30000, 30000],
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
    # 0 : [True, True],
    1 : [True, False],
    # 2 : [False, True],
    # 3 : [False, False],
    }

sparsity_pattern = (
    "ratio_based",
    # "n_per_line",
    )

gen_sparse_matrix = True


# Right now I'm relying on the secret invariant that only varmatmul_func
# uses sparse_fmts at all
def run_one_test(fname, m, n, sparsity, sparse_pattern, check, sparse_fmt_name):
    # m, n = size[0], size[1]
    f = functions[fname]
    sparse_fmt = sparsity_format[sparse_fmt_name]

    # We split appart the print because f(sparse_m, v) has the side effect of printing some data
    # And we want the newline to come after that printed stuff
    print("{}, {}, {}, {}, {}, ".format(str(fname), str(m), str(n), str(sparsity), sparse_pattern), end='')
    # Handling for sparse comparisons
    start = time.time()
    coords, vals, dense_shape, v = generate_sparse_coordinates_and_dense_v([m,n], sparsity)
    after_generation = time.time()
    print("{}, ".format(after_generation-start), end="")

    if sparse_fmt:
        # print(sparse_fmt)
        # print("started test")
        result, times, mid, memory_usage = f(coords, vals, dense_shape, v, sparse_fmt)
        # print("finished test")
    else:
        result, mid = f(coords, vals, dense_shape, v)
    after_computation = time.time()



    # We might need padding for the stuff spit out by the function
    if ("varmatmul_sparse_func" in fname):
        for t in times.flatten()[:8]:
            print("{}, ".format(t), end='')
        print(", "*8, end='')
        print("{}, ".format(sparse_fmt_name), end="")
        print("{}, ".format(after_computation-after_generation), end="")
        print("{}, ".format(0), end="")
        memory_usage = len(coords)*2*4+len(vals)*1*8+len(v)*8
        print("{}".format(memory_usage), end="\n")
    elif ("split" in fname):
        # print(len(times))
        for t in times[:16]:
            print("{}, ".format(t), end='')
        print("{}, ".format(sparse_fmt_name), end="")
        print("{}, ".format(mid-after_generation), end="")
        print("{}, ".format(after_computation-mid), end="")
        print("{}".format(memory_usage), end="\n") 
    else:
        print(", "*16, end='')
        print("{}, ".format(sparse_fmt_name), end="")
        print("{}, ".format(mid-after_generation), end="")
        print("{}, ".format(after_computation-mid), end="")
        memory_usage = len(coords)*2*4+len(vals)*1*8+len(v)*8
        print("{}".format(memory_usage), end="\n")


    if ('noop_func' not in fname) and check:
        good_result = tf_sparse_matmul_func(coords, vals, dense_shape, v)
        # print(result, good_result)
        # np.set_printoptions(threshold=np.nan)
        # print("allclose?: ", np.allclose(result, good_result))
        np.testing.assert_allclose(result, good_result)


def call_run_one_test(options):
    call(["python", "trivial_sparse.py", "test_one"] + [str(opt) for opt in options])
    time.sleep(0.3)


# Runs all combinations of the above variables.
# Puts the result into a csv for manipulation and graph generation
# in jupyter on the local machine.
def test_all(check):
    global gen_sparse_matrix
    print("function_type, m, n, sparsity, sparsity_pattern, generation_time, ", end='')
    print("TF_setup, tacotensor_init, tacotensor_insert, B_pack, c_pack, t_compile, t_assemble, t_compute, ", end='')
    print("TF_setup2, tacotensor_init2, tacotensor_insert2, B_pack2, c_pack2, t_compile2, t_assemble2, t_compute2, format, " +
          "computation_time, computation_time2, " + 
          "memory_usage")
    for fname, f in functions.iteritems():
        for (m, n) in sizes:
            for sparsity in sparsities:
                for sparse_pattern in sparsity_pattern:
                    if "varmatmul_sparse_func" in fname or "split" in fname:
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
        test_all(True)
        time.sleep(0.3)
        sys.stdout.flush()
    elif sys.argv[1] == "test_one":
        # print("about to call one test with args = ", *sys.argv[2:9])
        fname           = sys.argv[2]
        m               = int(sys.argv[3])
        n               = int(sys.argv[4])
        sparsity        = float(sys.argv[5])
        sparse_pattern  = sys.argv[6]
        check           = True if sys.argv[7]=="True" else False
        sparse_fmt      = int(sys.argv[8])
        run_one_test(fname, m, n, sparsity, sparse_pattern, check, sparse_fmt)







