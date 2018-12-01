from __future__ import division

import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import resampy
from scipy.io import wavfile
import glob
import mel_features
from datetime import datetime
from random import shuffle
# import vggish_input
import vggish_params
import vggish_slim

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_integer(
    'num_batches', 30,
    'Number of batches of examples to feed into the model. Each batch is of '
    'variable size and contains shuffled examples of each class of audio.')

flags.DEFINE_boolean(
    'train_vggish', True,
    'If True, allow VGGish parameters to change during training, thus '
    'fine-tuning VGGish. If False, VGGish parameters are fixed, thus using '
    'VGGish as a fixed feature extractor.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

FLAGS = flags.FLAGS

_NUM_CLASSES = 2 


### Defining Directories
pos_dir = './Dataset/balanced/gunshots/'
neg_dir = './Dataset/balanced/negative/'
# eval_pos_dir = 'Dataset/evaluation/gunshots/'
# eval_neg_dir = 'Dataset/evaluation/negative/' 
checkpoint_dir = './trained_checkpoint/'
# result_dir = 'results/'


def wavfile_to_examples(wav_file):
  
	sample_rate, wav_data = wavfile.read(wav_file)
	assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
	data = wav_data / 32768.0 # Convert to [-1.0, +1.0]

	# Convert to mono.
	if len(data.shape) > 1:
		data = np.mean(data, axis=1)
	# Resample to the rate assumed by VGGish.
	if sample_rate != vggish_params.SAMPLE_RATE:
		data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

	# Compute log mel spectrogram features.
	log_mel = mel_features.log_mel_spectrogram(data,
												audio_sample_rate= vggish_params.SAMPLE_RATE,
												log_offset= vggish_params.LOG_OFFSET,
												window_length_secs= vggish_params.STFT_WINDOW_LENGTH_SECONDS,
												hop_length_secs= vggish_params.STFT_HOP_LENGTH_SECONDS,
												num_mel_bins= vggish_params.NUM_MEL_BINS,
												lower_edge_hertz= vggish_params.MEL_MIN_HZ,
												upper_edge_hertz= vggish_params.MEL_MAX_HZ)

	# Frame features into examples.
	features_sample_rate = 1.0 /  vggish_params.STFT_HOP_LENGTH_SECONDS
	example_window_length = int(round( vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
	example_hop_length = int(round( vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
	log_mel_examples = mel_features.frame(log_mel,
											window_length=example_window_length,
											hop_length=example_hop_length)

	return log_mel_examples


def get_examples_batch(pos_data_file, neg_data_file):
	"""Returns a shuffled batch of examples of all audio classes.
	Note that this is just a toy function because this is a simple demo intended
	to illustrate how the training code might work.
	Returns:
	a tuple (features, labels) where features is a NumPy array of shape
	[batch_size, num_frames, num_bands] where the batch_size is variable and
	each row is a log mel spectrogram patch of shape [num_frames, num_bands]
	suitable for feeding VGGish, while labels is a NumPy array of shape
	[batch_size, num_classes] where each row is a multi-hot label vector that
	provides the labels for corresponding rows in features.
	"""

	# Make examples of each signal and corresponding labels.
	# Sine is class index 0, Const class index 1, Noise class index 2.
	pos_examples = wavfile_to_examples(pos_data_file)
	pos_labels = np.array([[1, 0]] * pos_examples.shape[0])
	neg_examples = wavfile_to_examples(neg_data_file)
	neg_labels = np.array([[0, 1]] * neg_examples.shape[0])


	# Shuffle (example, label) pairs across all classes.
	all_examples = np.concatenate((pos_examples, neg_examples))
	all_labels = np.concatenate((pos_labels, neg_labels))
	labeled_examples = list(zip(all_examples, all_labels))
	shuffle(labeled_examples)

	# Separate and return the features and labels.
	features = [example for (example, _) in labeled_examples]
	labels = [label for (_, label) in labeled_examples]

	return (features, labels)


def main(_):
	with tf.Graph().as_default(), tf.Session() as sess:
	# Define VGGish.
		embeddings = vggish_slim.define_vggish_slim(FLAGS.train_vggish)

	# Define a shallow classification model and associated training ops on top
	# of VGGish.
		with tf.variable_scope('mymodel'):
			# Add a fully connected layer with 100 units.
			num_units = 100
			fc = slim.fully_connected(embeddings, num_units)

			# Add a classifier layer at the end, consisting of parallel logistic
			# classifiers, one per class. This allows for multi-class tasks.
			logits = slim.fully_connected(
			  fc, _NUM_CLASSES, activation_fn=None, scope='logits')
			tf.sigmoid(logits, name='prediction')

			# Add training ops.
			with tf.variable_scope('train'):

				# global_step = tf.Variable(04, name='global_step', trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

		    	# Labels are assumed to be fed as a batch multi-hot vectors, with
		    	# a 1 in the position of each positive class label, and 0 elsewhere.
				labels = tf.placeholder(tf.float32, shape=(None, _NUM_CLASSES), name='labels')

				# Cross-entropy label loss.
				xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels, name='xent')
				loss = tf.reduce_mean(xent, name='loss_op')
				tf.summary.scalar('loss', loss)

				# We use the same optimizer and hyperparameters as used to train VGGish.
				optimizer = tf.train.AdamOptimizer(learning_rate=vggish_params.LEARNING_RATE, epsilon=vggish_params.ADAM_EPSILON)
				optimizer.minimize(loss, name='train_op')

			# Initialize all variables in the model, and then load the pre-trained
			# VGGish checkpoint.
			sess.run(tf.global_variables_initializer())
			# vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)

			## Defining checkpoint file 
			saver = tf.train.Saver()

			#Saving the checkpoint file
			ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
			if ckpt:
				print('loaded ' + ckpt.model_checkpoint_path)
				saver.restore(sess, ckpt.model_checkpoint_path)

			# Locate all the tensors and ops we need for the training loop.
			features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
			labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
			# global_step_tensor = sess.graph.get_tensor_by_name('mymodel/train/global_step:0')
			loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
			train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')

			train_iter = len(glob.glob(pos_dir + '*.wav'))
			print(train_iter)

			step = 0

			#Defining loss summary variable
			loss_summary = tf.summary.scalar(name='lossSummary', tensor=loss_tensor)

			#Defining writer to save logs in a log directory
			log_path_train = 'logdir' + '/train_{}'.format(datetime.utcnow().strftime("%Y%m%d%H%M%S"))
			train_writer = tf.summary.FileWriter(log_path_train, sess.graph)
			summaries_train = tf.summary.merge_all()

			# print('got to the training loop')
			# The training loop.
			for epoch in range(0,500):
				# print('inside the training loop')
				for ind in range(train_iter):
					# print('Step', step)
					pos_path = (glob.glob(pos_dir + '*.wav'))
					neg_path = (glob.glob(neg_dir + '*.wav'))

					(features, labels) = get_examples_batch(pos_path[ind], neg_path[ind])

					[summary, loss, _] = sess.run([summaries_train, loss_tensor, train_op], feed_dict={features_tensor: features, labels_tensor: labels})
					
					if step % 10 == 0:
						train_writer.add_summary(summary, global_step=step)
						print("Epoch: {0} - Loss: {1}".format(epoch, loss))

					step += 1
				saver.save(sess, checkpoint_dir + 'model.ckpt')

			train_writer.close()
			print("Training Ended: Checkpoint File and Summary Logs Saved")

if __name__ == '__main__':
  tf.app.run()




























