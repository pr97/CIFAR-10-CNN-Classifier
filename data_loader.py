from pickle_wrapper import read_pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_processing import *


def get_label_names():
	obj = read_pickle("batches.meta")
	obj = obj[b"label_names"]
	return obj

def load_training_batches():

	"""
	Loads training data into a python dictionary called batches.
	batches: python_dict containing batches of data under key 'batch_i' (ith batch)
				   Each batch is a dictionary with the actual CIFAR-10 image data
				   under key b'data':-
			batches['batch_i'][b'data'] -- a 10000x3072 numpy array of uint8s. 
			Each row of the array stores a 32x32 colour image. 
			The first 1024 entries contain the red channel values, 
			the next 1024 the green, and the final 1024 the blue. 
			The image is stored in row-major order, so that the first
			32 entries of the array are the red channel values of the first row of the image.

	Returns:
		batches

	"""

	batches = {}
	for i in range(1, 6):
		file_name = "data_batch_" + str(i)
		batches["batch_" + str(i)] = read_pickle(file_name)

	return batches

def load_test_data():
	"""
	Loads test data into a python dictionary called test_set.
	
	test_set: python_dict containing test_data under key b'data'
		test_set[b'data'] -- a 10000x3072 numpy array of uint8s. 
		Each row of the array stores a 32x32 colour image. 
		The first 1024 entries contain the red channel values, 
		the next 1024 the green, and the final 1024 the blue. 
		The image is stored in row-major order, so that the first
		32 entries of the array are the red channel values of the first row of the image.

	Returns:
		test_set
	"""
	test_set = read_pickle("test_batch")
	return test_set

def main():
	batch = "batch_3"
	eg_no = 676

	label_names = get_label_names()
	print(label_names)

	train_data_orig = load_training_batches()
	
	x = process_batches(train_data_orig)
	X = x[batch]
	Y = process_labels(train_data_orig)

	print(label_names[int(Y[batch][eg_no])])

	plt.imshow(X[eg_no])
	plt.show()

	test_data_orig = load_test_data()
	test_data_vec = test_data_orig[b'data']
	test_data_labels = test_data_orig[b'labels']
	print(test_data_vec.shape)

	test_data_img = vec2img(test_data_vec)
	print(test_data_img.shape)
	print("+++++++++++++++++++++++++++++++++++")
	print(type(test_data_labels))
	print("+++++++++++++++++++++++++++++++++++")

	one_hot = generate_one_hot(test_data_labels, 10)

	with tf.Session() as sess:
		tf.global_variables_initializer()
		print(one_hot.shape)
		print(sess.run(one_hot[0:5]))

if __name__ == '__main__':
	main()