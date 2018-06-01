import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_loader import *

def process_batches(data_dict, num_batches = 5):

	"""
	Params:
		data_dict: python_dict containing batches of data under key 'batch_i' (ith batch)
				   Each batch is a dictionary with the actual CIFAR-10 image data
				   under key b'data':-
			data_dict['batch_i'][b'data] -- a 10000x3072 numpy array of uint8s. 
			Each row of the array stores a 32x32 colour image. 
			The first 1024 entries contain the red channel values, 
			the next 1024 the green, and the final 1024 the blue. 
			The image is stored in row-major order, so that the first
			32 entries of the array are the red channel values of the first row of the image.

		num_batches: int stating number of batches of data in data_dict. Defaults to 5. 

	Returns:
		processed_dict: python_dict containing batches of data under key 'batch_i' (ith batch)
						Each batch is a numpy ndarray of shape = (10000, 32, 32, 3) with 
						actual CIFAR-10 image data, formated in "NHWC" format, suitable being 
						fed into a CNN.
	"""

	processed_dict= {}
	for i in range(1, num_batches + 1):
		x = data_dict["batch_" + str(i)][b'data']
		num_eg = data_dict["batch_" + str(i)][b'data'].shape[0]
		x = x.reshape((num_eg, 32, 32, 3), order = 'F')
		x = np.rot90(x, -1, (1, 2))
		processed_dict["batch_" + str(i)] = x

	return processed_dict

def vec2img(vec):
	"""
	Converts vector to rgb image while preserving the 'num_examples' dimension.
	Assumes the first dimension to be the 'num_batches' dimension.
	"""

	num_eg = vec.shape[0]
	img = vec.reshape((num_eg, 32, 32, 3), order = 'F')
	img = np.rot90(img, -1, (1, 2))

	return img

def process_test_labels(test_data_orig_dict):
	return test_data_orig_dict[b'labels']


def process_labels(data_dict, num_batches = 5):
	"""
	Params:
		data_dict: python_dict containing batches of labels under key 'batch_i' (ith batch)
				   Each batch is a dictionary with the actual CIFAR-10 image labels
				   under key b'labels':-
			data_dict['batch_i'][b'labels'] -- a 10000x1 numpy array of uint8s. 
			Each row of the array stores a label from 0 to 9 (inclusive) denoting the image category. 
			

		num_batches: int stating number of batches of data in data_dict. Defaults to 5. 

	Returns:
		processed_dict: python_dict containing batches of labels under key 'batch_i' (ith batch)
						Each batch is a numpy ndarray of shape = (10000) with the actual
						CIFAR-10 image labels from 0 to 9 (inclusive) denoting the image category.
	"""

	processed_dict= {}

	for i in range(1, num_batches + 1):
		y = data_dict["batch_" + str(i)][b'labels']
		shape = len(data_dict["batch_" + str(i)][b'labels'])
		y = np.array(y, dtype = np.int32).reshape((shape))
		processed_dict["batch_" + str(i)] = y

	return processed_dict

def generate_one_hot(labels, num_classes):
	one_hot_tensor = tf.one_hot(indices = labels, depth = num_classes, axis = -1)
	return one_hot_tensor

def data_normalizer(data, normalizer = "for_rgb_image"):
	if normalizer == "for_rgb_image":
		return data / 255.



def main():
	train_data_orig = load_training_batches()
	batch = "batch_1"
	eg_no = 0
	x = process_batches(train_data_orig)
	X = x[batch]
	y = process_labels(train_data_orig)
	Y = y[batch]
	one_hot = generate_one_hot(Y, 10)

	with tf.Session() as sess:
		tf.global_variables_initializer()
		print(one_hot.shape)
		print(sess.run(one_hot[0:5]))

if __name__ == '__main__':
	main()