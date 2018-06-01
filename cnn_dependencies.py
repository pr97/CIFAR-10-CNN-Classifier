import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from math import floor

# from matplotlib import style

# style.use("ggplot")

def create_placeholders(n_H0, n_W0, n_C0, n_y):
	"""
	Params:
		m - number of training examples in batch
		n_H - image height (in px)
		n_W - image width (in px)
		n_C - number of channels in image (defaults to 3 for "rgb")

	Returns:
		X, Y - Tensorflow placeholders for cnn input and labels.
	"""

	X = tf.placeholder(dtype = tf.float32, shape = [None, n_H0, n_W0, n_C0], name = 'X')
	Y = tf.placeholder(dtype = tf.float32, shape = [None, n_y], name = 'Y')

	return X, Y

def init_params(n_H0, n_W0, n_C0, n_y):
	"""
	Modify this function to accept a general cnn architecture.
	"""
	
	params = {}

	#CONV 1
	n_C1 = 96
	f1 = 7	#filter_size
	s1 = 1	#stride
	p1 = 2	#padding
	W1 = tf.get_variable(dtype = tf.float32,
		shape = [f1, f1, n_C0, n_C1],
		initializer = tf.contrib.layers.xavier_initializer(),
		name = "W1")
	n_H1 = floor((n_H0 + 2 * p1 - f1) / s1) + 1 #28
	n_W1 = floor((n_W0 + 2 * p1 - f1) / s1) + 1 #28

	#CONV2
	n_C2 = 96
	f2 = 4	#filter_size
	s2 = 1	#stride
	p2 = 1	#padding
	W2 = tf.get_variable(dtype = tf.float32,
		shape = [f2, f2, n_C1, n_C2],
		initializer = tf.contrib.layers.xavier_initializer(),
		name = "W2")
	
	n_H2 = floor((n_H1 + 2 * p2 - f2) / s2) + 1 #27
	n_W2 = floor((n_W1 + 2 * p2 - f2) / s2) + 1 #27

	#CONV 3 ("SAME")
	n_C3 = 256
	f3 = 5	#filter_size
	s3 = 1	#stride
	p3 = (f3 - 1) / 2	#padding for SAME CONV
	W3 = tf.get_variable(dtype = tf.float32,
		shape = [f3, f3, n_C2, n_C3],
		initializer = tf.contrib.layers.xavier_initializer(),
		name = 'W3')
	
	n_H3 = floor((n_H2 + 2 * p3 - f3) / s3) + 1 #27
	n_W3 = floor((n_W2 + 2 * p3 - f3) / s3) + 1 #27
	

	#POOL 3
	f_p3 = 3
	s_p3 = 2
	p_p3 = 0
	n_H3 = floor((n_H3 - f_p3) / s_p3) + 1 #13
	n_W3 = floor((n_W3 - f_p3) / s_p3) + 1 #13

	#CONV 4 ("SAME")
	n_C4 = 384
	f4 = 3	#filter_size
	s4 = 1	#stride
	p4 = (f4 - 1) / 2	#padding for SAME CONV
	W4 = tf.get_variable(dtype = tf.float32,
		shape = [f4, f4, n_C3, n_C4],
		initializer = tf.contrib.layers.xavier_initializer(),
		name = 'W4')
	
	n_H4 = floor((n_H3 + 2 * p4 - f4) / s4) + 1 #13
	n_W4 = floor((n_W3 + 2 * p4 - f4) / s4) + 1 #13

	#CONV 5 ("SAME")
	n_C5 = 384
	f5 = 3	#filter_size
	s5 = 1	#stride
	p5 = (f5 - 1) / 2	#padding for SAME CONV
	W5 = tf.get_variable(dtype = tf.float32,
		shape = [f5, f5, n_C4, n_C5],
		initializer = tf.contrib.layers.xavier_initializer(),
		name = "W5")
	
	n_H5 = n_H4	#SAME CONV (13)
	n_W5 = n_W4	#SAME CONV (13)

	#CONV 6 ("SAME")
	n_C6 = 256
	f6 = 3	#filter_size
	s6 = 1	#stride
	p6 = (f6 - 1) / 2	#padding for SAME CONV
	W6 = tf.get_variable(dtype = tf.float32,
		shape = [f6, f6, n_C5, n_C6],
		initializer = tf.contrib.layers.xavier_initializer(),
		name = "W6")
	
	n_H6 = n_H5	#SAME CONV (13)
	n_W6 = n_W5	#SAME CONV (13)

	#POOL 6
	f_p6 = 3
	s_p6 = 2
	p_p6 = 0
	n_H6 = floor((n_H6 - f_p6) / s_p6) + 1
	n_W6 = floor((n_W6 - f_p6) / s_p6) + 1

	"""
	INPUT FOR FC LAYERS: 6x6x256
	"""
	params["W1"] = W1
	params["W2"] = W2
	params["W3"] = W3
	params["W4"] = W4
	params["W5"] = W5
	params["W6"] = W6

	architecture_hparams = {"CONV1":(s1, p1),
	"CONV2":(s2, p2),
	"CONV3":(s3, p3),
	"POOL3":(f_p3, s_p3, p_p3),
	"CONV4":(s4, p4),
	"CONV5":(s5, p5),
	"CONV6":(s6, p6),
	"POOL6":(f_p6, s_p6, p_p6)}

	return params, architecture_hparams

def forward_prop(X_train, params, architecture_hparams):
	
	#Retrieve model parameters from the params dict.
	W1 = params["W1"]
	W2 = params["W2"]
	W3 = params["W3"]
	W4 = params["W4"]
	W5 = params["W5"]
	W6 = params["W6"]

	#Retrieve model architecture hyper-parameters from the architecture_hparams dict.
	s1, p1 = architecture_hparams["CONV1"]
	s2, p2 = architecture_hparams["CONV2"]
	s3, p3 = architecture_hparams["CONV3"]
	f_p3, s_p3, p_p3 = architecture_hparams["POOL3"]
	s4, p4 = architecture_hparams["CONV4"]
	s5, p5 = architecture_hparams["CONV5"]
	s6, p6 = architecture_hparams["CONV6"]
	f_p6, s_p6, p_p6 = architecture_hparams["POOL6"]

	A0 = tf.pad(tensor = X_train, paddings = tf.constant([[0, 0], [p1, p1], [p1, p1], [0, 0]]), name = 'A0')
	Z1 = tf.nn.conv2d(input = A0, filter = W1, strides = [1, s1, s1, 1], padding = "VALID", data_format = "NHWC", name = "Z1")
	A1 = tf.nn.relu(Z1, name = "A1")
	# print(A1.shape)

	A1 = tf.pad(tensor = A1, paddings = tf.constant([[0, 0], [p2, p2], [p2, p2], [0, 0]]))
	# print("++++++++++++++++++++++++++++++++")
	# print(A1.shape)
	# print(W2.shape)
	# print("++++++++++++++++++++++++++++++++")
	Z2 = tf.nn.conv2d(input = A1, filter = W2, strides = [1, s2, s2, 1], padding = "VALID", data_format = "NHWC", name = "Z2")
	A2 = tf.nn.relu(Z2, name = "A2")

	Z3 = tf.nn.conv2d(input = A2, filter = W3, strides = [1, s3, s3, 1], padding = "SAME", data_format = "NHWC", name = "Z3")
	A3 = tf.nn.relu(Z3, name = "A3")
	P3 = tf.nn.max_pool(value = A3, ksize = [1, f_p3, f_p3, 1], strides = [1, s_p3, s_p3, 1], padding = "VALID", data_format = "NHWC", name = "P3")

	Z4 = tf.nn.conv2d(input = P3, filter = W4, strides = [1, s4, s4, 1], padding = "SAME", data_format = "NHWC", name = "Z4")
	A4 = tf.nn.relu(Z4, name = "A4")

	Z5 = tf.nn.conv2d(input = A4, filter = W5, strides = [1, s5, s5, 1], padding = "SAME", data_format = "NHWC", name = "Z5")
	A5 = tf.nn.relu(Z5, name = "A5")

	Z6 = tf.nn.conv2d(input = A5, filter = W6, strides = [1, s6, s6, 1], padding = "SAME", data_format = "NHWC", name = "Z6")
	A6 = tf.nn.relu(Z6, name = "Z6")
	P6 = tf.nn.max_pool(value = A6, ksize = [1, f_p6, f_p6, 1], strides = [1, s_p6, s_p6, 1], padding = "VALID", data_format = "NHWC", name = "P6")
	
	P6_flat = tf.contrib.layers.flatten(inputs = P6)

	Z7 = tf.contrib.layers.fully_connected(inputs = P6_flat, num_outputs = 4096)
	A7 = tf.nn.relu(Z7, name = "A7")	

	Z8 = tf.contrib.layers.fully_connected(inputs = A7, num_outputs = 10)

	return Z8

def softmax_cost(Z8, Y_train):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z8, labels = Y_train))
	return cost


def main():
	pass

if __name__ == '__main__':
	main()





























	


def main():
	pass

if __name__ == '__main__':
	main()