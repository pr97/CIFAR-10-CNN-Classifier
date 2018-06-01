import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

from data_loader import *
from data_processing import *
from cnn_dependencies_v2 import *

from datetime import datetime

def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """
    vector = np.int32(vector)
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return np.float32(result)

def model(X_train, Y_train, X_test, Y_test, num_classes = 10, learning_rate = 0.001, optimization = "adam", num_epochs = 1000):
	"""
	X_train - python dict containing minibatches of train data (numpy array) under key "batch_i". shape of each minibatch - (10000, 32, 32, 3)
	Y_train - python dict containing minibatches of train data labels (0 to 9) (numpy array) under key "batch_i". shape of each minibatch - (10000,)
	X_test - numpy array containing pre-processed test images. shape - (10000, 32, 32, 3)
	Y_test - python list (compatible with tf.one_hot function) containing test labels (0 to 9). shape - (10000,)
	"""

	X_train_minibatches = [np.float32(X_train[key]) for key in X_train.keys()]
	Y_train_minibatches = [np.float32(Y_train[key]) for key in X_train.keys()]

	###########################
	for i in range(len(Y_train_minibatches)):
		Y_train_minibatches[i] = convertToOneHot(Y_train_minibatches[i], num_classes = 10)
	###########################

	assert(X_train_minibatches[0].shape == (10000, 32, 32, 3))

	for i in range(len(X_train_minibatches)):
		X_train_minibatches[i] = data_normalizer(X_train_minibatches[i])

	print(type(X_train_minibatches))
	for minibatch in X_train_minibatches:
		print(type(minibatch))

	X_train_tensors = []
	for minibatch in X_train_minibatches:
		X_train_tensors.append(tf.constant(minibatch, dtype = tf.float32) / tf.constant(255., dtype = tf.float32))
	# Y_train_tensors = []
	# for minibatch in X_train_minibatches:
	# 	X_train_tensors.append(tf.constant(minibatch))
	# X_train_tensors = [tf.constant(minibatch) for minibatch in X_train_minibatches]
	Y_train_tensors = [generate_one_hot(minibatch, num_classes) for minibatch in Y_train_minibatches]
	assert(len(X_train_tensors) == len(Y_train_tensors))

	X_test_tensor = tf.constant(np.float32(X_test)) / tf.constant(255., dtype = tf.float32)
	Y_test_tensor = generate_one_hot(Y_test, num_classes)

	Y_test_one_hot = convertToOneHot(Y_test, num_classes=10)

	minibatch_size, n_H0, n_W0, n_C0 = X_train_minibatches[0].shape

	X, Y = create_placeholders(n_H0, n_W0, n_C0, num_classes)

	params, architecture_hparams = init_params(n_H0, n_W0, n_C0, num_classes)

	Z8 = forward_prop(X, params, architecture_hparams)

	cost = softmax_cost(Z8, Y)

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	saver = tf.train.Saver()
	# boilerplate for tensor-board logging
	now = datetime.utcnow().strftime("%Y-%m-%d--%H-%M-%S")
	root_logdir = "tf-logs"
	logdir = "C:/Users/praty/Desktop/DL/CIFAR - 10/" + "{}/run-{}/".format(root_logdir, now)
	cost_summary = tf.summary.scalar('softmax_cost', cost)
	file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
	######################################
	

	with tf.Session() as sess:
		# saver.restore(sess, "C:/Users/praty/Desktop/DL/CIFAR - 10/model_params/model.ckpt")
		sess.run(init)

		# for epoch in range(num_epochs):
		# 	epoch_cost = 0
		# 	num_minibatches = len(X_train_tensors)

		# 	for i in range(len(X_train_tensors)):
		# 		X_train_tensor = X_train_tensors[i]
		# 		Y_train_tensor = Y_train_tensors[i]

		# 		_, curr_cost = sess.run([optimizer, cost], feed_dict = {X: X_train_tensor, Y: Y_train_tensor})

		# 		epoch_cost += curr_cost / num_minibatches

		# 	if epoch % 1 == 0:
		# 		print("Epoch[" + str(epoch) + "] cost = " + str(epoch_cost))

		for epoch in range(num_epochs):
			epoch_cost = 0
			num_minibatches = len(X_train_tensors)

			for i in range(len(X_train_tensors)):
				X_train_tensor = X_train_minibatches[i]
				Y_train_tensor = Y_train_minibatches[i]
				chunk_cost = 0
				for j in range(100):
					X_train_chunk = X_train_tensor[(j * 100):((j + 1) * 100)]
					Y_train_chunk = Y_train_tensor[(j * 100):((j + 1) * 100)]
					# label_names = get_label_names()
					# print(label_names[int(np.argmax(Y_train_chunk[53]))])
					# plt.imshow(X_train_chunk[53])
					# plt.show()
					_, curr_cost = sess.run([optimizer, cost], feed_dict = {X: X_train_chunk, Y: Y_train_chunk})
					chunk_cost += curr_cost / 100.
					print("Chunk " + str(j) + ", batch " + str(i) + " Epoch " + str(epoch) + " cost = " + str(curr_cost))
					# if (j == 2 or j == 3) and i == 0 and epoch == 0:
					# 	saver.save(sess, "C:/Users/praty/Desktop/DL/CIFAR - 10/model_params/model.ckpt")
					if j == 100:
						summary_str = cost_summary.eval({X: X_train_chunk, Y: Y_train_chunk})
						step = epoch * num_minibatches + i * 100 + j
						file_writer.add_summary(summary_str, step)

				epoch_cost += chunk_cost / num_minibatches

			if epoch % 1 == 0:
				saver.save(sess, "C:/Users/praty/Desktop/DL/CIFAR - 10/model_params/model.ckpt")

		saver.save(sess, "C:/Users/praty/Desktop/DL/CIFAR - 10/model_params/model.ckpt")

		train_accuracy = 0
		for i in range(len(X_train_tensors)):
			X_train_tensor = X_train_minibatches[i][0:100]
			Y_train_tensor = Y_train_minibatches[i][0:100]

			prediction = tf.argmax(Z8, 1)
			label_prediction = tf.argmax(Y, 1)

			is_correct = tf.equal(prediction, label_prediction)

			minibatch_accuracy = tf.reduce_mean(tf.float32(is_correct))
			train_accuracy += minibatch_accuracy.eval({X:X_train_tensor, Y:Y_train_tensor}) / len(X_train_tensors)

		test_accuracy = 0
		prediction = tf.argmax(Z8, 1)
		label_prediction = tf.argmax(Y, 1)
		is_correct = tf.equal(prediction, label_prediction)
		# test_accuracy = tf.reduce_mean(tf.float32(is_correct)).eval({X:X_test_tensor, Y:Y_test_tensor})
		test_accuracy = tf.reduce_mean(tf.float32(is_correct)).eval({X:X_test, Y:Y_test_one_hot})

		print("Train Accuracy = " + str(train_accuracy))
		print("Test Accuracy = " + str(test_accuracy))

	file_writer.close()



	
def main():
	

	trian_data_orig = load_training_batches()
	X_processed_train = process_batches(trian_data_orig)
	Y_processed_train = process_labels(trian_data_orig)

	test_data_orig = load_test_data()
	test_data_vec = test_data_orig[b'data']
	test_data_labels = test_data_orig[b'labels']
	X_processed_test = vec2img(test_data_vec)
	Y_processed_test = test_data_labels
	model(X_train = X_processed_train,
		Y_train = Y_processed_train,
		X_test = X_processed_test,
		Y_test = Y_processed_test,
		num_classes = 10,
		learning_rate = 0.01,
		optimization = "adam",
		num_epochs = 1000)

if __name__ == '__main__':
	main()