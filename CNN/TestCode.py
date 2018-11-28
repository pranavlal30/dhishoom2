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
import vggish_postprocess
import vggish_params
import vggish_slim


### Defining Directories
# pos_dir = './Dataset/balanced/gunshots/'
# neg_dir = './Dataset/balanced/negative/'
eval_pos_dir = 'Dataset/evaluation/gunshots/'
eval_neg_dir = 'Dataset/evaluation/negative/' 
checkpoint_dir = 'trained_checkpoint/'
pca_params = 'vggish_pca_params.npz'

_NUM_CLASSES = 2 
rel_error = 0.1

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


def resize(data, axis, new_size):
    shape = list(data.shape)

    pad_shape = shape[:]
    pad_shape[axis] = np.maximum(0, new_size - shape[axis])

    shape[axis] = np.minimum(shape[axis], new_size)
    shape = np.stack(shape)

    slices = [slice(0, s) for s in shape]

    resized = np.concatenate([
      data[slices],
      np.zeros(np.stack(pad_shape))
    ], axis)

    # Update shape.
    new_shape = list(data.shape)
    new_shape[axis] = new_size
    resized.reshape(new_shape)
    return resized

def get_features(sess, data, processorObject):

	pproc = processorObject
	features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
	embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

	[embedding_batch] = sess.run([embedding_tensor],feed_dict={features_tensor: data})
	postprocessed_batch = pproc.postprocess(embedding_batch)

	return postprocessed_batch


# def process_features(sess, features):

# 	num_frames = np.minimum(features.shape[0], vggish_params.MAX_FRAMES)
# 	data = resize(features, 0, vggish_params.MAX_FRAMES)
# 	data = np.expand_dims(data, 0)
# 	num_frames = np.expand_dims(num_frames, 0)

# 	input_tensor = sess.graph.get_collection("input_batch_raw")[0]
# 	print(input_tensor.shape)
# 	num_frames_tensor = sess.graph.get_collection("num_frames")
# 	predictions_tensor = sess.graph.get_collection("predictions")

# 	predictions_val, = sess.run([predictions_tensor],feed_dict={input_tensor: data, num_frames_tensor: num_frames })

# 	return predictions_val

# def get_predictions(sess, data):
# 	# samples = data / 32768.0  # Convert to [-1.0, +1.0]
# 	examples_batch = wavfile_to_examples(data)
# 	features = get_features(sess, examples_batch)
# 	# predictions = process_features(sess, features)

# 	return features

# def run():

graph = tf.Graph()

with graph.as_default():
	sess = tf.Session()
	vggish_slim.define_vggish_slim(training=False)
	vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_dir)
	pproc = vggish_postprocess.Postprocessor(pca_params)

	eval_iter = len(glob.glob(eval_pos_dir + '*.wav'))

	for ind in range(eval_iter):

		pos_path = (glob.glob(eval_pos_dir + '*.wav'))
		neg_path = (glob.glob(eval_neg_dir + '*.wav'))

		# print(type(pos_path[ind]))
		# print(neg_path[ind])
		pos_features = wavfile_to_examples(pos_path[ind])
		neg_features = wavfile_to_examples(neg_path[ind])

		# print()
		# print('Gunshot Predictions', get_features(sess, pos_features, pproc))
		# print('Negative Predictions', get_features(sess, neg_features, pproc))


		postprocessed_batch = get_features(sess, pos_features, pproc)
		print('Postprocessed VGGish embedding: ', postprocessed_batch[0])
		# Write the postprocessed embeddings as a SequenceExample, in a similar
		# format as the features released in AudioSet. Each row of the batch of
		# embeddings corresponds to roughly a second of audio (96 10ms frames), and
		# the rows are written as a sequence of bytes-valued features, where each
		# feature value contains the 128 bytes of the whitened quantized embedding.
		# seq_example = tf.train.SequenceExample(feature_lists=tf.train.FeatureLists(
		# 									feature_list={vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:
		# 									tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[embedding.tobytes()]))
		# 									for embedding in postprocessed_batch])}))

		# print(seq_example)

		# expected_postprocessed_mean = 123.0
		# expected_postprocessed_std = 75.0
		# np.testing.assert_allclose([np.mean(postprocessed_batch), np.std(postprocessed_batch)],
		# 							[expected_postprocessed_mean, expected_postprocessed_std],
		# 							rtol=rel_error)








