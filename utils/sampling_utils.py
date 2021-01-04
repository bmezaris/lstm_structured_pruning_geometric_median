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
"""Contains a collection of util functions for model construction."""
import tensorflow as tf

def SampleRandomSequence(model_input, num_frames, num_samples):
  """Samples a random sequence of frames of size num_samples (that is from original frames, not the padded frames).

  Args:
    model_input: A tensor of size batch_size x max_frames x feature_size
    num_frames: A tensor of size batch_size x 1
    num_samples: A scalar

  Returns:
    `model_input`: A tensor of size batch_size x num_samples x feature_size
  """

  batch_size = tf.shape(model_input)[0]
  frame_index_offset = tf.tile(tf.expand_dims(tf.range(num_samples), 0),
                               [batch_size, 1])
  max_start_frame_index = tf.math.maximum(num_frames - num_samples, 0)
  start_frame_index = tf.cast(
      tf.multiply(tf.random.uniform([batch_size, 1]), tf.cast(max_start_frame_index + 1, tf.float32)),
      tf.int32)
  frame_index = tf.math.minimum(start_frame_index + frame_index_offset,
                           tf.cast(num_frames - 1, tf.int32))
  batch_index = tf.tile(tf.expand_dims(tf.range(batch_size), 1),
                        [1, num_samples])
  index = tf.stack([batch_index, frame_index], 2)
  return tf.gather_nd(model_input, index)