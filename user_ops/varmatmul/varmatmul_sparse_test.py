import tensorflow as tf
import numpy as np

# mymatmul_module = tf.load_op_library('/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/mymatmul.so')
varmatmul_module = tf.load_op_library('/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/varmatmul.so')
varmatmul_sparse_module = tf.load_op_library('/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/varmatmul_sparse.so')
loader_module = tf.load_op_library('/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/loader.so')
spmv_taco_input_module = tf.load_op_library('/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/spmv_taco_input.so')


sess = tf.Session('')
with sess:
    # a = [[1,2],[3,4]]
    # b = [[1],[2]]
    a = np.random.randint(5, size=(100, 100))
    b = np.random.randint(5, size=(100, 1))
    a = np.array(a).astype(np.float64)
    b = np.array(b).astype(np.float64)

    indices = np.argwhere(a)
    values = a[zip(*indices)].astype(np.float64)
    dense_shape = np.array(a.shape)
    indices = indices.astype(np.int64)

    indices = tf.cast(indices, tf.int64)
    values = tf.cast(values, tf.double)
    dense_shape = tf.cast(dense_shape, tf.int64)
    b = tf.cast(b, tf.double)

    # print indices.dtype, values.dtype, dense_shape.dtype, b.dtype
    # print mymatmul_module.my_matmul(a, b).eval()
    result, times = sess.run(varmatmul_module.var_matmul(a, b, sparse_fmt=[True, False]))
    print result, times[:10]
    # print "indices: ", indices, "values: ", values, "dense_shape: ", dense_shape, "b: ", b
    # o1, o2, o3, o4 = sess.run(varmatmul_sparse_module.var_matmul_sparse(indices, values, dense_shape, b, sparse_fmt=[True, False]))
    o1, o2, o3, o4, times_ = loader_module.rob_loader(indices, values, dense_shape, b, sparse_fmt=[True, False])
    # print "1:  ", o1, "2: ", o2, "3: ", o3, "4: ", o4, "b: ", b
    # print times[:10]
    result_, times2_ = spmv_taco_input_module.spmv_taco_input(o1, o2, o3, o4, b, sparse_fmt=[True, False])
    result2, times, times2 = sess.run([result_, times_, times2_])
    print result2, times, times2
    # result, times = sess.run(varmatmul_sparse_module.var_matmul_sparse(indices, values, dense_shape, b, sparse_fmt=[True, False]))
    
    np.testing.assert_array_equal(result, result2)




    # Not working...
    # print varmatmul_module.var_matmul(a, b).eval()