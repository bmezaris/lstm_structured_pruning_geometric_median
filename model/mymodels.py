""""
Implementation in Tensorflow of the BLSTM model described in:
N. Gkalelis, V. Mezaris, "Structured Pruning of LSTMs via
Eigenanalysis and Geometric Median for Mobile Multimedia
and Deep Learning Applications", Proc. 22nd IEEE Int.
Symposium on Multimedia (ISM), Dec. 2020.
History
-------
DATE      | DESCRIPTION    | NAME              | Organization |
1/07/2020 | first creation | Nikolaos Gkalelis | CERTH-ITI    |
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

class MyLstm(tf.keras.Model):
    def __init__(self, num_classes,
                 weight_decay = 0.005,
                 sequence_length = 300,
                 lstm_size = 1024,
                 video_feature_size = 1024,
                 audio_feature_size = 128):
        super(MyLstm, self).__init__()

        total_feature_size = video_feature_size + audio_feature_size
        input_shape = (sequence_length, total_feature_size)
        lstm_fw = tf.keras.layers.LSTM(int(lstm_size / 2), return_sequences=True)
        lstm_bw = tf.keras.layers.LSTM(int(lstm_size / 2), return_sequences=True, go_backwards=True)
        self.bdr = tf.keras.layers.Bidirectional(lstm_fw, backward_layer=lstm_bw, input_shape=input_shape)
        self.lstm = tf.keras.layers.LSTM(int(lstm_size))
        self.dns = tf.keras.layers.Dense(units=num_classes, activation='sigmoid',
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay))


    def call(self, inputs):

        x1 = self.bdr(inputs)
        x2 = self.lstm(x1)
        x = self.dns(x2)

        return x, x1, x2
