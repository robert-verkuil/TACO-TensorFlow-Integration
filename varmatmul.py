from __future__ import print_function
import tensorflow as tf
import numpy as np
import timeit
import time

varmatmul_module = tf.load_op_library(
	'/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/user_ops/varmatmul.so')

  # acceptable fill methods are
  # d, u, r, s, h, v, l, f, b

  # If a tns_file is not specified, uses first to parameters as sparse_m and v.
  # Otherwise, it uses the tns file and a generated vector dertermined by v_fmt_str
def SpMV(fmt, tns_file="", v_fmt_str=""):
	with tf.Session(''):
		order = len(fmt)
		sparse_m = np.array([[17., 2.], [3., 4.]]).astype(np.float64)
		v = np.array([[5.], [7.]]).astype(np.float64)
		varmatmul_module.var_matmul(sparse_m, v, sparse_fmt=fmt, tns_file=tns_file, v_fmt_str=v_fmt_str).eval()

def foo(fmt, tns_file):
    def _foo():
    	SpMV(fmt, tns_file=tns_path+tns_file, v_fmt_str="d")
    return _foo

tns_path = "/home/ubuntu/tensorflow/tensorflow/core/user_ops/varmatmul/"
def loadTNS(filename, dense_shape, flatten):
	filepath = tns_path + filename

	indices = []
	values = []
	dense_shape = []

	for line in open(filepath):
		if line.strip():
			# data = map(long, line.split())
			data = map(long, line.split()[:-1]) + [long(float(line.split()[-1]))]
			order = len(data) - 1

			coords = data[:order]
			val = data[order]
			# If we have to flatten the tensor so that it has order 2
			# if flatten:
			# 	coords[0] *= 
			# 	coords


			indices.append(coords)
			values.append(val)

	return indices, values

# TODO: do I need different formats?
def TfSpMV(tns_file="", dense_shape = []):
	with tf.Session(''):
		# if tns_file == "":
		# 	sparse_m = np.array([[17., 2.], [3., 4.]]).astype(np.float64)
		# 	v = np.array([[5.], [7.]]).astype(np.float64)
		# else:
		# load tns_file into tf.SparseTensor
		indices, values = loadTNS(tns_file, dense_shape, True)
		print("len(indices)=", len(indices), "len(values)=", len(values), "dense_shape=", dense_shape)
		sparse_m = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
		v = np.random.randint(10, size=(dense_shape[-1], 1))
		tf.sparse_tensor_dense_matmul(sparse_m, tf.cast(v, tf.int32)).eval()


def TestAllTaco(use_taco):
	# order4_fmts = [[False, False, False, False]]
	# order5_fmts = [[False, False, False, False, False]]
	print("tns file, A_fmt, B_fmt, c_fmt, A_dim, B_dim, c_dim, B_pack, c_pack, t_compile, t_assemble, t_compute")
	for tns_file, order, dense_shape, fmt in [
							# ("nips.tns", 4, [2482, 2862, 14036, 17], [False, False, False, False]), 
							("nipsTrivial.tns", 2, [5, 2158], [False, False]), 
							("nips2D.tns", 2, [2483, 2863], [False, False]), 
							("vast2D.tns", 2, [165427, 11374], [False, False]), 
							# ("delicious-4d.tns", 4)
							]:
		# for fmt in (order4_fmts if order==4 else order5_fmts):
			print("\"" + tns_file + "\", ", end='')
			if use_taco:
				SpMV(fmt, tns_file=tns_path+tns_file, v_fmt_str="d")
			else:
				TfSpMV(tns_file, dense_shape)

if __name__ == '__main__':
	TestAllTaco(True)
	# TestAllTaco(False)


# FROSTT Tensors used:
# http://frostt.io/tensors/nips/
# NIPS Publications
# Non-zeros	3,101,609
# Order	4
# Dimensions	2,482 x 2,862 x 14,036 x 17
# nips.tns

# http://frostt.io/tensors/vast-2015-mc1/
# VAST 2015 Mini-Challenge 1
# Non-zeros	26,021,945
# Order	5
# Dimensions	165,427 x 11,374 x 2 x 100 x 89
# vast-2015-mc1-5d.tns

# http://frostt.io/tensors/lbnl-network/
# LBNL-Network
# Non-zeros	1,698,825
# Order	5
# Dimensions	1,605 x 4,198 x 1,631 x 4,209 x 868,131
# lbnl-network.tns

# http://frostt.io/tensors/delicious/
# Delicious
# Non-zeros	140,126,181
# Order	4
# Dimensions	532,924 x 17,262,471 x 2,480,308 x 1,443
# delicious-4d.tns

# http://frostt.io/tensors/amazon-reviews/ IGNORED FOR NOW!
# Amazon Reviews
# Non-zeros	1,741,809,018
# Order	3
# Dimensions	4,821,207 x 1,774,269 x 1,805,187
# None
