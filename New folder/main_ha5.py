###
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import libraries
import numpy as np
import scipy
import matplotlib.pyplot as plt
from keras.datasets import mnist
from util import func_confusion_matrix

# my imports
import tensorflow as tf
import time
from datetime import datetime
import os.path
import func_two_layer_fc
import math
from PIL import Image
#from my_util import load_data, gen_batch

import math
from data_load_helper import loadData, load_data, gen_batch
# ! my imports


# load (downloaded if needed) the MNIST dataset
x, y = loadData()
spliter = math.floor(len(x)*2/3)

x_train = x[:spliter]
y_train = y[:spliter]

x_test = x[spliter:]
y_test = y[spliter:]


#(x_train, y_train), (x_test, y_test) = mnist.load_data()


# transform each image from 28 by28 to a 784 pixel vector
# pixel_count = x_train.shape[1] * x_train.shape[2]
# x_train = x_train.reshape(x_train.shape[0], pixel_count).astype('float32')
# x_test = x_test.reshape(x_test.shape[0], pixel_count).astype('float32')

# normalize inputs from gray scale of 0-255 to values between 0-1
x_train = x_train
x_test = x_test


spliter = math.floor(len(x_train)*2/3)
x_validate = x_train[spliter:]
y_validate = y_train[spliter:]
x_train = x_train[:spliter]
y_train = y_train[:spliter]

####################################################################
############## 								My Code       		####################
####################################################################
# Model parameters as external flags
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.05, 'Learning rate for the training.')
flags.DEFINE_integer('max_steps', 200, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('batch_size', 400,
  'Batch size. Must divide dataset sizes without remainder.')
flags.DEFINE_string('train_dir', 'tf_logs',
  'Directory to put the training data.')
flags.DEFINE_float('reg_constant', 0.1, 'Regularization constant.')

FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
  print('{} = {}'.format(attr, value))
print()

IMAGE_PIXELS = len(x[0])
print(IMAGE_PIXELS)
CLASSES = 1

beginTime = time.time()

####################################################################
############## 					step-1: load data          #################
####################################################################

# Put logs for each run in separate directory
logdir = FLAGS.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'

# Uncommenting these lines removes randomness
# You'll get exactly the same result on each run
# np.random.seed(1)
# tf.set_random_seed(1)

# Load CIFAR-10 data
data_sets = load_data(x_train, y_train, x_validate, y_validate)
#print(data_sets)

####################################################################
############## step-2: Prepare the Tensorflow graph ################ 
####################################################################

# -----------------------------------------------------------------------------
# Prepare the Tensorflow graph
# (We're only defining the graph here, no actual calculations taking place)
# -----------------------------------------------------------------------------

# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS],
  name='images')
labels_placeholder = tf.placeholder(tf.int64, shape=[None], name='image-labels')

# Operation for the classifier's result
logits = func_two_layer_fc.inference(images_placeholder, IMAGE_PIXELS,
  FLAGS.hidden1, CLASSES, reg_constant=FLAGS.reg_constant)

# Operation for the loss function
loss = func_two_layer_fc.loss(logits, labels_placeholder)

# Operation for the training step
train_step = func_two_layer_fc.training(loss, FLAGS.learning_rate)

# Operation calculating the accuracy of our predictions
accuracy = func_two_layer_fc.evaluation(logits, labels_placeholder)

# Operation merging summary data for TensorBoard
summary = tf.summary.merge_all()

# Define saver to save model state at checkpoints
saver = tf.train.Saver()

# -----------------------------------------------------------------------------
# Run the TensorFlow graph
# -----------------------------------------------------------------------------
with tf.Session() as sess:
	# Initialize variables and create summary-writer
	sess.run(tf.global_variables_initializer())
	summary_writer = tf.summary.FileWriter(logdir, sess.graph)

	# Generate input data batches
	zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
	batches = gen_batch(list(zipped_data), FLAGS.batch_size, FLAGS.max_steps)

	for i in range(FLAGS.max_steps):

		# Get next input data batch
		batch = next(batches)
		images_batch, labels_batch = zip(*batch)
		feed_dict = {
			images_placeholder: images_batch,
			labels_placeholder: labels_batch
		}

		# Periodically print out the model's current accuracy
		#if i % 100 == 0:
			#train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
			#print('Step {:d}, training accuracy {:g}'.format(i, train_accuracy))
			#summary_str = sess.run(summary, feed_dict=feed_dict)
			#summary_writer.add_summary(summary_str, i)

		# Perform a single training step
		sess.run([train_step, loss], feed_dict=feed_dict)

		# Periodically save checkpoint
		if (i + 1) % 1000 == 0:
			checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
			saver.save(sess, checkpoint_file, global_step=i)
			print('Saved checkpoint')

	# After finishing the training, evaluate on the test set
	# test_accuracy = sess.run(accuracy, feed_dict={
		# images_placeholder: data_sets['images_test'],
		# labels_placeholder: data_sets['labels_test']})
	# print('Test accuracy {:g}'.format(test_accuracy))

	
		
	# get confusion matrix
	# output = sess.run(tf.argmax(logits, 1), {images_placeholder: data_sets['images_test']})
	# cm, acc, arrR, arrP = func_confusion_matrix(data_sets['labels_test'], output)
	# print('\n#####################################\n')
	# print('Confusion matrix is :\n', cm)
	# print('Average accuracy is :', acc)
	# print('Precision rate for each class is \n:', arrR)
	# print('Recall rate for each class is \n:', arrP)
	# print('\n#####################################\n')
	# ! get confusion matrix
	
	# get confusion matrix
	# output = sess.run(tf.argmax(logits, 1), {images_placeholder: x_test})
	# cm, acc, arrR, arrP = func_confusion_matrix(y_test, output)
	# print('\n#####################################\n')
	# print('Confusion matrix is :\n', cm)
	# print('Average accuracy is :', acc)
	# print('Precision rate for each class is \n:', arrR)
	# print('Recall rate for each class is \n:', arrP)
	# print('\n#####################################\n')
	# ! get confusion matrix
	
	
	# visualization error
	# for i,d in enumerate(output):
		# counter = 0
		# if output[i] != y_test[i]:
			# if counter < 10: 
				# print(output[i], y_test[i])
				# arr = [abs(math.floor(x * 255)) for x in x_test[i]]
				# arr = np.array(arr)
				# arr.resize((28,28))
				# im = Image.fromarray(arr)
				# im.show()
				# var = input("Would you like to pop next error image: ")
				# counter += 1

	# ! visualization error
	
endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))