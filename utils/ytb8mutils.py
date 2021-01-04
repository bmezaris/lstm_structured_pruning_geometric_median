# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Functions from the YouTube-8M Tensorflow Starter Code.
Code obtained from:
https://github.com/google/youtube-8m/blob/master/
"""

import tensorflow as tf

### utilties to read frame level tfrecords ###

def resize_axis(tensor, axis, new_size, fill_value=0):
  """Truncates or pads a tensor to new_size on on a given axis.

  Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
  size increases, the padding will be performed at the end, using fill_value.

  Args:
    tensor: The tensor to be resized.
    axis: An integer representing the dimension to be sliced.
    new_size: An integer or 0d tensor representing the new value for
      tensor.shape[axis].
    fill_value: Value to use to fill any new entries in the tensor. Will be cast
      to the type of tensor.

  Returns:
    The resized tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.unstack(tf.shape(tensor))

  pad_shape = shape[:]
  pad_shape[axis] = tf.math.maximum(0, new_size - shape[axis])

  shape[axis] = tf.math.minimum(shape[axis], new_size)
  shape = tf.stack(shape)

  resized = tf.concat([
      tf.slice(tensor, tf.zeros_like(shape), shape),
      tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
  ], axis)

  # Update shape.
  new_shape = tensor.get_shape().as_list()  # A copy is being made.
  new_shape[axis] = new_size
  resized.set_shape(new_shape)
  return resized

def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
  """Dequantize the feature from the byte format to the float format.

  Args:
    feat_vector: the input 1-d vector.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

  Returns:
    A float vector which has the same shape as feat_vector.
  """
  assert max_quantized_value > min_quantized_value
  quantized_range = max_quantized_value - min_quantized_value
  scalar = quantized_range / 255.0
  bias = (quantized_range / 512.0) + min_quantized_value
  return feat_vector * scalar + bias

def get_video_matrix(features, feature_size, max_frames,
                     max_quantized_value, min_quantized_value):
  """Decodes features from an input string and quantizes it.

  Args:
    features: raw feature values
    feature_size: length of each frame feature vector
    max_frames: number of frames (rows) in the output feature_matrix
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

  Returns:
    feature_matrix: matrix of all frame-features
    num_frames: number of frames in the sequence
  """

  decoded_features = tf.reshape(
    tf.cast(tf.io.decode_raw(input_bytes=features, out_type=tf.uint8), tf.float32),
    [-1, feature_size])

  num_frames = tf.math.minimum(tf.shape(decoded_features)[0], max_frames)
  feature_matrix = Dequantize(decoded_features, max_quantized_value,
                                    min_quantized_value)
  feature_matrix = resize_axis(feature_matrix, 0, max_frames)
  return feature_matrix, num_frames

def parse_yt8m_tfrecs_seq_ex_func(sequence_example_proto):
  # sequence_example_proto: serialized sequence example proto

  num_classes = 3862
  feature_names = ["rgb", "audio"]
  feature_sizes = [1024, 128]
  max_frames = 300
  max_quantized_value = 2
  min_quantized_value = -2

  # features dicts
  context_features = { "id": tf.io.FixedLenFeature([], tf.string),
                       "labels": tf.io.VarLenFeature(tf.int64)}
  sequence_features = {
    feature_name: tf.io.FixedLenSequenceFeature([], dtype=tf.string)
    for feature_name in feature_names
  }

  # parse the input tf.Example proto (protocol) using the dictionary feature maps above
  contexts, features = tf.io.parse_single_sequence_example(
    serialized=sequence_example_proto,
    context_features=context_features,
    sequence_features=sequence_features)

  # loads (potentially) different types of features and concatenates them
  num_feature_types = len(feature_names)
  assert num_feature_types > 0, "No feature selected: feature_names is empty!"

  assert len(feature_names) == len(feature_sizes), (
    "length of feature_names (={}) != length of feature_sizes (={})".format(
      len(feature_names), len(feature_sizes)))

  num_frames = -1  # initialize the number of frames in the video
  feature_matrices = [None] * num_feature_types  # an array of different features
  for feature_index in range(num_feature_types):
    feature_matrix, num_frames_in_this_feature = get_video_matrix(
      features[feature_names[feature_index]],
      feature_sizes[feature_index], max_frames,
      max_quantized_value, min_quantized_value)
    if num_frames == -1:
      num_frames = num_frames_in_this_feature

    feature_matrices[feature_index] = feature_matrix

  # cap the number of frames at self.max_frames
  num_frames = tf.math.minimum(num_frames, max_frames)

  # concatenate different features
  video_matrix = tf.concat(feature_matrices, 1)

  # Process video-level labels.
  label_indices = contexts["labels"].values
  sparse_labels = tf.sparse.SparseTensor(
    tf.expand_dims(label_indices, axis=-1),
    tf.ones_like(contexts["labels"].values, dtype=tf.bool),
    (num_classes,))
  labels = tf.sparse.to_dense(sparse_labels,
                              default_value=False,
                              validate_indices=False)
  # convert to batch format.
  batch_video_ids = tf.expand_dims(contexts["id"], 0)
  batch_video_matrix = tf.expand_dims(video_matrix, 0)
  batch_labels = tf.expand_dims(labels, 0)
  batch_frames = tf.expand_dims(num_frames, 0)
  batch_label_weights = None

  output_dict = {
    "video_ids": batch_video_ids,
    "video_matrix": batch_video_matrix,
    "labels": batch_labels,
    "num_frames": batch_frames,
  }
  if batch_label_weights is not None:
    output_dict["label_weights"] = batch_label_weights

  return output_dict

### other utilties ###

def CrossEntropyLoss(labels, predictions):
  epsilon = 1e-8
  float_labels = tf.cast(labels, tf.float32)
  cross_entropy_loss = float_labels * tf.math.log(predictions + epsilon) + (
          1 - float_labels) * tf.math.log(1 - predictions + epsilon)
  cross_entropy_loss = tf.negative(cross_entropy_loss)
  return tf.reduce_sum(cross_entropy_loss, 1)
