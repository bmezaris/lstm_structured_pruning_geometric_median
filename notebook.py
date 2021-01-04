""""
Implementation in Tensorflow of the BLSTM structured pruning method
using the YouTube8M dataset, as described in the paper:
N. Gkalelis and V. Mezaris, “Fractional step discriminant pruning:
A filter pruning framework for deep convolutional neural networks,”
in IEEE ICMEW, London, UK, Jul. 2020, pp. 1–6.
History
-------
DATE      | DESCRIPTION    | NAME              | Organization |
1/07/2020 | first creation | Nikolaos Gkalelis | CERTH-ITI    |
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
from absl import flags
from absl import app

import numpy as np
import datetime as dtm
from time import process_time
from scipy.spatial import distance as scipyspatial_distance

# local imports
from utils.ytb8mutils import parse_yt8m_tfrecs_seq_ex_func
from utils.sampling_utils import SampleRandomSequence
from utils.ytb8mutils import CrossEntropyLoss
from utils.eval_util import GapScore
from model.mymodels import MyLstm as BlstmMdl



FLAGS = flags.FLAGS

flags.DEFINE_string("DBNM", r'.\dbs',"The directory where tfrecord files are.")
flags.DEFINE_string("train_data_pattern_glob", r'yt8m\tfrecords\frame\train\train*.tfrecord',
                    "File glob for the training dataset; relevant path to DBNM.")
flags.DEFINE_string("eval_data_pattern_glob", r'yt8m\tfrecords\frame\validate\validate*.tfrecord',
                    "File glob for the validation dataset; relevant path to DBNM.")
flags.DEFINE_bool("shuffle_data", True, "Shuffle the data on read.")
flags.DEFINE_integer("num_parallel_calls", 4, "Number of threads to use in map function when processing the dataset.")
flags.DEFINE_integer("num_classes", 3862, "Number of threads to use in map function when processing the dataset.") # 3862
flags.DEFINE_integer("num_train_observations", 3888919, "Number of training observations.") # 3888919, 41393
flags.DEFINE_integer("top_k", 20, "How many predictions to output per video.")

# Model flags.
flags.DEFINE_float("weight_decay", 0.005, "Weight decay.")
flags.DEFINE_integer("sequence_length", 300, "Number of frames in video sequence.")
flags.DEFINE_integer("lstm_size", 1024, "Number of LSTM nodes.")
flags.DEFINE_integer("video_feature_size", 1024, "Dimensionality of visual feature vectors.")
flags.DEFINE_integer("audio_feature_size", 128, "Dimensionality of audio feature vectors.")
flags.DEFINE_string("optimizer", "AdamOptimizer", "What optimizer class to use.")
flags.DEFINE_integer("batch_size", 160, "How many examples to process per batch for training.")
flags.DEFINE_integer("fea_vec_dim", 1152, "Feature vector dimensionality.")
flags.DEFINE_float("regularization_penalty", 1e-3, "How much weight to give to the regularization loss (the label loss has a weight of 1).")
flags.DEFINE_float("base_learning_rate", 0.0002, "Which learning rate to start with.")
flags.DEFINE_float("learning_rate_decay", 0.95, "Learning rate decay factor to be applied every learning_rate_decay_examples.")
flags.DEFINE_float("learning_rate_decay_examples", 4000000, "Multiply current learning rate by learning_rate_decay every learning_rate_decay_examples.") # 3888919 4000000
flags.DEFINE_integer("num_epochs", 20, "How many passes to make over the dataset before halting training.")
flags.DEFINE_float("step_prune", 200, "Number of steps to prune the network.")
flags.DEFINE_float("epoch_prune", 0, "Epoch to start pruning procedure.")
flags.DEFINE_float("keep_energy_rate", 0.9, "keep rate for energy preservation.")
flags.DEFINE_float("prune_rate_l1", 0.3, "prune rate for 1st lstm.")
flags.DEFINE_float("prune_rate_l2", 0.3, "prune rate for 2nd lstm.")
flags.DEFINE_float("prune_rate_l3", 0.3, "prune rate for 3rd lstm.")

def main(unused_argv):
    DBNM = FLAGS.DBNM
    #print(DBNM)
    TARGET_PRUNE_RATE = np.mean([FLAGS.prune_rate_l1, FLAGS.prune_rate_l2, FLAGS.prune_rate_l3])

    model_save_path = os.path.join(os.getcwd(), 'checkpoints')  # Path to save trained models
    print('Model save path: {}'.format(model_save_path))

    experiment_trace_filename = 'eval_pr' + str(TARGET_PRUNE_RATE) + '_' \
                                + dtm.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.txt'
    fw = open(experiment_trace_filename, 'w')

    train_data_pattern = os.path.join(FLAGS.DBNM, FLAGS.train_data_pattern_glob) # absolute file glob for the dataset
    eval_data_pattern = os.path.join(FLAGS.DBNM, FLAGS.eval_data_pattern_glob) # absolute file glob for the validation dataset

    shuffle_buffer_size = FLAGS.batch_size *2 # 5
    # tfrecordFilesTrn = tf.data.Dataset.list_files(file_pattern=FLAGS.train_input_data_pattern_glob, shuffle=False)
    tfrecordFnamesTrn = tf.io.gfile.glob(train_data_pattern)
    DstTrn = tf.data.TFRecordDataset(tfrecordFnamesTrn)  # create dataset object for this tfrecord file
    DstTrn = DstTrn.map(map_func=parse_yt8m_tfrecs_seq_ex_func, num_parallel_calls=FLAGS.num_parallel_calls)
    DstTrn = DstTrn.shuffle(buffer_size=shuffle_buffer_size)
    DstTrn = DstTrn.batch(FLAGS.batch_size)

    tfrecordFnamesTst = tf.io.gfile.glob(eval_data_pattern)
    DstTst = tf.data.TFRecordDataset(tfrecordFnamesTst)  # create dataset object for this tfrecord file
    DstTst = DstTst.map(map_func=parse_yt8m_tfrecs_seq_ex_func, num_parallel_calls=FLAGS.num_parallel_calls)
    DstTst = DstTst.batch(FLAGS.batch_size)

    model = BlstmMdl(num_classes=FLAGS.num_classes,
                    weight_decay=FLAGS.weight_decay,
                    sequence_length=FLAGS.sequence_length,
                    lstm_size=FLAGS.lstm_size,
                    video_feature_size=FLAGS.video_feature_size,
                    audio_feature_size=FLAGS.audio_feature_size)

    loss_object = CrossEntropyLoss

    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=FLAGS.base_learning_rate,
        decay_steps=FLAGS.learning_rate_decay_examples,
        decay_rate=FLAGS.learning_rate_decay,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate_fn)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    @tf.function
    def train_step(fea, labels):
        with tf.GradientTape() as tape:
            predictions, x1, x2 = model(fea, training=True)
            labels_batch_loss = loss_object(labels, predictions)
            loss = tf.reduce_mean(labels_batch_loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        return x1, x2

    @tf.function
    def test_step(fea, labels):
        predictions, _, _ = model(fea, training=False)
        labels_batch_loss = loss_object(labels, predictions)
        loss = tf.reduce_mean(labels_batch_loss)
        test_loss(loss)

        return predictions, labels, labels_batch_loss

    # function to select the cells to prune
    def selectCellsToPruneGm(W, num_cell_prune):
        # W: H x 4*F + 4*H

        # compute importance score for each cell
        distmat = scipyspatial_distance.cdist(W, W, 'euclidean')  # lstm cell distance matrix
        hta = np.sum(np.abs(distmat), axis=0)  # lstm cell importance score
        cell_prune_index = hta.argsort()[:num_cell_prune] # get the ones with smaller importance score

        # LSTM cell indicator vector
        H = W.shape[0] # number of cells (hidden states)
        lstmPruneIndicator = np.zeros((H,), dtype=np.bool)  # initialize indicator vector (indicating cells to prune)
        lstmPruneIndicator[cell_prune_index] = True

        return lstmPruneIndicator

    # model pruning function
    def prune_model(prune_rates=(0.1, 0.1, 0.1)):

        lstmPruneIndicatorFw = []
        lstmPruneIndicatorBw = []
        num_cell_prune_fw = []
        num_cell_prune_bw = []

        # first prune bidirectional layer
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Bidirectional):
                W = layer.get_weights()
                F = W[0].shape[0] # input space dimensionality
                H = W[1].shape[0] # hidden vector dimensionality (number of cells)

                num_cell_prune_fw = int(prune_rates[0] * H) # number of cells to prune

                # prune forward LSTM in Bidirectional LSTM
                lstmPruneIndicatorFw = selectCellsToPruneGm(
                    np.vstack(np.hsplit(W[0], 4) + np.hsplit(W[1], 4)).transpose(), num_cell_prune_fw)

                lstmPruneIndicatorFwExt = np.tile(lstmPruneIndicatorFw, (4,))  # extended along all LSTM matrices

                # prune selected cells
                W[0][:, lstmPruneIndicatorFwExt] = np.zeros((F, 4 * num_cell_prune_fw))
                W[1][:, lstmPruneIndicatorFwExt] = np.zeros((H, 4 * num_cell_prune_fw))
                W[1][lstmPruneIndicatorFw, :] = np.zeros((num_cell_prune_fw, 4 * H))

                # prune backward LSTM of Bidirectional LSTM
                num_cell_prune_bw = int(prune_rates[1] * H)  # number of cells to prune
                lstmPruneIndicatorBw = selectCellsToPruneGm(
                    np.vstack(np.hsplit(W[3], 4) + np.hsplit(W[4], 4)).transpose(), num_cell_prune_bw)

                lstmPruneIndicatorBwExt = np.tile(lstmPruneIndicatorBw, (4,))  # extended along all LSTM matrices

                # prune selected cells
                W[3][:, lstmPruneIndicatorBwExt] = np.zeros((F, 4 * num_cell_prune_bw))
                W[4][:, lstmPruneIndicatorBwExt] = np.zeros((H, 4 * num_cell_prune_bw))
                W[4][lstmPruneIndicatorBw, :] = np.zeros((num_cell_prune_bw, 4 * H))

                # reset LSTM layer weights
                layer.set_weights(W)

        lstmPruneIndicatorBd = tf.concat(values=[lstmPruneIndicatorFw, lstmPruneIndicatorBw], axis=0)
        num_cell_prune_bd = num_cell_prune_fw + num_cell_prune_bw

        # next prune 2nd LSTM layer
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.LSTM):
                W = layer.get_weights()
                F = W[0].shape[0]  # input space dimensionality
                H = W[1].shape[0]  # hidden vector dimensionality (number of cells)

                num_cell_prune_lstm = int(prune_rates[2] * H)  # number of cells to prune
                lstmPruneIndicatorLstm = selectCellsToPruneGm(
                    np.vstack(np.hsplit(W[0], 4) + np.hsplit(W[1], 4)).transpose(), num_cell_prune_lstm)

                lstmPruneIndicatorLstmExt = np.tile(lstmPruneIndicatorLstm, (4,))  # extended along all LSTM matrices

                # prune cells that correspond to hidden states of bidirectional lstm
                W[0][lstmPruneIndicatorBd,:] = np.zeros((num_cell_prune_bd, 4 * H))

                # prune selected cells
                W[0][:, lstmPruneIndicatorLstmExt] = np.zeros((F, 4 * num_cell_prune_lstm))
                W[1][:, lstmPruneIndicatorLstmExt] = np.zeros((H, 4 * num_cell_prune_lstm))
                W[1][lstmPruneIndicatorLstm, :] = np.zeros((num_cell_prune_lstm, 4 * H))

                # reset LSTM layer weights
                layer.set_weights(W)

    # compute sample mean and covariance matrix recursiverly
    def cmpt_covmat_recursv(m, S, N, xx):
        mm = tf.reduce_mean(xx, axis=0, keepdims=True)
        SS = tf.linalg.matmul(xx, xx, transpose_a=True)
        NN = xx.shape[0]

        # update
        NNN = NN + N
        m = (N / NNN) * m + (NN / NNN) * mm
        S = (N / NNN) * S + (NN / NNN) * SS
        N = NNN

        return m, S, N

    # compute prune rate using covarinace matrix
    def cmpt_prune_rate(S):
        S = S + tf.transpose(S)
        evl, _ = tf.linalg.eigh(S)
        evl = evl.numpy()
        evl[evl < 0.] = 0.  # remove eigenvalues smaller than 0: covariance matrix is always positive semi-definite
        evl = evl / np.sum(evl)  # normalize to sum to 1
        evl[::-1].sort()
        evlcs = np.cumsum(evl, axis=0)  # cumulative sum
        keep_cells = np.sum(evlcs > FLAGS.keep_energy_rate)  # keep cells that correspond to keep_energy_rate

        return keep_cells


    ls1 = FLAGS.lstm_size // 2 # number of cells of forward branch in bidirectional lstm
    ls2 =  FLAGS.lstm_size // 2 # number of cells of backward branch in bidirectional lstm
    ls3 =  FLAGS.lstm_size # number of cells of lstm after bidirectional lstm
    for epoch in range(FLAGS.num_epochs):

        # initialize one covariance matrix and mean vector for each LSTM
        S1 = tf.zeros((ls1, ls1))
        S2 = tf.zeros((ls2, ls2))
        S3 = tf.zeros((ls3, ls3))
        m1 = tf.zeros((1,ls1))
        m2 = tf.zeros((1,ls2))
        m3 = tf.zeros((1,ls3))
        N = 0 # total number of observations

        train_loss.reset_states() # Reset the metrics at the start of the next epoch
        test_loss.reset_states()

        # training
        t0 = 0.  # model training time
        t1 = 0.  # model pruning time
        steptrn = 1
        num_samples = 300
        for viddata in DstTrn: # fea_rgb, fea_audio, labels, vidid
            #vidid = viddata["video_ids"]
            feaSeq = viddata["video_matrix"]
            labels = viddata["labels"]
            num_frames = viddata["num_frames"]
            feaSeq = tf.squeeze(feaSeq, axis=[1]) # or feaSeq[:, 0, :, :]
            labels = tf.squeeze(labels, axis=[1])
            num_frames = tf.cast(num_frames, tf.float32)
            feaSeq = SampleRandomSequence(feaSeq, num_frames, num_samples)
            feature_dim = len(feaSeq.get_shape()) - 1
            feaSeq = tf.nn.l2_normalize(feaSeq, feature_dim)
            t0_start = process_time()
            x1, xx3 = train_step(feaSeq, labels)
            x1 = x1[:, -1, :]
            xx1 = x1[:, 0:ls1]
            xx2 = x1[:, ls1:ls1 + ls2]

            m1, S1, NNN = cmpt_covmat_recursv(m1, S1, N, xx1)
            m2, S2, _ = cmpt_covmat_recursv(m2, S2, N, xx2)
            m3, S3, _ = cmpt_covmat_recursv(m3, S3, N, xx3)
            S1 = S1 - tf.linalg.matmul(m1, m1, transpose_a=True)
            S2 = S2 - tf.linalg.matmul(m2, m2, transpose_a=True)
            S3 = S3 - tf.linalg.matmul(m3, m3, transpose_a=True)
            N = NNN

            t0_stop = process_time()
            t0 += t0_stop - t0_start

            if steptrn % 100 == 0:
                print("Train step: {}; Loss {}".format(steptrn, train_loss.result()))
                arecord = 'Train step: {}; Model train time (total):  {}; Loss: {}'
                jrecord = arecord.format(steptrn, t0, train_loss.result())
                print(jrecord)
                with open(experiment_trace_filename, "a") as text_file:
                    print(jrecord, file=text_file)

            if (steptrn % FLAGS.step_prune) == 0 and (epoch >=  FLAGS.epoch_prune):
                print("Pruning model with pruning rate {}".format(TARGET_PRUNE_RATE))
                t1_start = process_time()
                prnTol = 0.01

                prune_rate_diff = 2* prnTol # initialize to something larger
                stepprntune = 0
                suggested_total_prune_rate = 0.
                suggested_prune_rate_l1, suggested_prune_rate_l2, suggested_prune_rate_l3 = 0., 0., 0.
                while abs(prune_rate_diff) > prnTol and FLAGS.keep_energy_rate < 0.9999999 and stepprntune < 150:

                    keep_cells_l1 = cmpt_prune_rate(S1)
                    keep_cells_l2 = cmpt_prune_rate(S2)
                    keep_cells_l3 = cmpt_prune_rate(S3)

                    suggested_prune_rate_l1 = 1. - keep_cells_l1 / ls1
                    suggested_prune_rate_l2 = 1. - keep_cells_l2 / ls2
                    suggested_prune_rate_l3 = 1. - keep_cells_l3 / ls3

                    suggested_total_prune_rate = np.mean([suggested_prune_rate_l1, suggested_prune_rate_l2, suggested_prune_rate_l3])
                    prune_rate_diff = TARGET_PRUNE_RATE - suggested_total_prune_rate
                    if prune_rate_diff > 0: # still prune rate is small; keep more energy rate to reduce prune rate towards the target one
                        FLAGS.keep_energy_rate = FLAGS.keep_energy_rate + prune_rate_diff * (1- FLAGS.keep_energy_rate)
                    elif prune_rate_diff < 0: # prune rate is large; keep less energy rate to increase pruning rate towards the target one
                        FLAGS.keep_energy_rate = FLAGS.keep_energy_rate + prune_rate_diff * (FLAGS.keep_energy_rate / 4.)

                    print('Tune step: {}; Keep energy rate: {}; keeping cells: l1: {}, l2: {}, l3: {}'.format(stepprntune, FLAGS.keep_energy_rate, keep_cells_l1, keep_cells_l2, keep_cells_l3))
                    print('Tune step: {}; Keep energy rate: {}; Suggested pruning rates: total: {}, l1: {}, l2: {}, l3: {}'.format(stepprntune, FLAGS.keep_energy_rate, suggested_total_prune_rate, suggested_prune_rate_l1, suggested_prune_rate_l2, suggested_prune_rate_l3))
                    stepprntune = stepprntune +1

                # many eigenvlues may be zero, i.e., small redudancy in the layers; then we need to enforce larger pruning rates
                if abs(prune_rate_diff) > prnTol:
                    print('Scaling pruning rates to adjust to target pruning rate...')
                    a1 = TARGET_PRUNE_RATE / suggested_total_prune_rate
                    suggested_prune_rate_l1 = a1 * suggested_prune_rate_l1
                    suggested_prune_rate_l2 = a1 * suggested_prune_rate_l2
                    suggested_prune_rate_l3 = a1 * suggested_prune_rate_l3

                    suggested_total_prune_rate = np.mean(
                        [suggested_prune_rate_l1, suggested_prune_rate_l2, suggested_prune_rate_l3])

                    print('Adjusted pruning rates with adjusting ratio {}'.format(a1))

                FLAGS.prune_rate_l1 = suggested_prune_rate_l1
                FLAGS.prune_rate_l2 = suggested_prune_rate_l2
                FLAGS.prune_rate_l3 = suggested_prune_rate_l3

                print( 'Total prune rate: {}; Individual pruning rates: l1: {}, l2: {}, l3: {}'.format(
                    suggested_total_prune_rate, FLAGS.prune_rate_l1, FLAGS.prune_rate_l2, FLAGS.prune_rate_l3))


                prune_model(prune_rates = (FLAGS.prune_rate_l1, FLAGS.prune_rate_l2, FLAGS.prune_rate_l3))
                t1_stop = process_time()
                t1 += t1_stop - t1_start
                arecord = 'Train step: {}; Model pruning time (total):  {}, Rate: {}'
                jrecord = arecord.format(steptrn, t1, TARGET_PRUNE_RATE)
                print(jrecord)
                with open(experiment_trace_filename, "a") as text_file:
                    print(jrecord, file=text_file)


            steptrn += 1

        # prune model at the end of the epoch prior evaluation
        prune_model(prune_rates = (FLAGS.prune_rate_l1, FLAGS.prune_rate_l2, FLAGS.prune_rate_l3))
        print("Total train steps: {}".format(steptrn))

        # save model
        model_save_pathname = os.path.join(model_save_path, 'eval_pr' + str(TARGET_PRUNE_RATE)
                                           + dtm.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        print('Model save pathname: {}'.format(model_save_pathname))
        model.save_weights(model_save_pathname)

        # testing
        evl_metrics = GapScore(num_class=FLAGS.num_classes, top_k=FLAGS.top_k)

        steptst = 0
        t2 = 0.  # model testing time
        for viddata in DstTst:
            #vidid = viddata["video_ids"]
            feaSeq = viddata["video_matrix"]
            labels = viddata["labels"]
            num_frames = viddata["num_frames"]
            feaSeq = tf.squeeze(feaSeq, axis=[1])
            labels = tf.squeeze(labels, axis=[1])
            num_frames = tf.cast(num_frames, tf.float32)
            feaSeq = SampleRandomSequence(feaSeq, num_frames, num_samples)
            feature_dim = len(feaSeq.get_shape()) - 1
            feaSeq = tf.nn.l2_normalize(feaSeq, feature_dim)

            t2_start = process_time()
            test_pred, test_labels, test_loss_batch = test_step(feaSeq, labels)
            t2_stop = process_time()
            t2 += t2_stop - t2_start

            test_pred = test_pred.numpy()
            test_labels = test_labels.numpy()
            #test_loss_batch = test_loss_batch.numpy()

            # retain necessary information for performance evaluation
            _ = evl_metrics.accumulate(test_pred, test_labels)

            if steptst % 200 == 0:
                print("Testing step {}".format(steptst))
                arecord = 'Test step: {}; Model test  time (total):  {}; Loss:  {}'
                jrecord = arecord.format(steptst, t2, test_loss.result())
                print(jrecord)
                with open(experiment_trace_filename, "a") as text_file:
                    print(jrecord, file=text_file)

            steptst += 1

        gap = evl_metrics.get()
        print("Total testing steps {}".format(steptst))
        arecord = 'Epoch {}, Loss: {}, Test Loss: {}, GAP: {}'
        jrecord = arecord.format(epoch, train_loss.result(), test_loss.result(), gap)
        print(jrecord)
        with open(experiment_trace_filename, "a") as text_file:
            print(jrecord, file=text_file)

    fw.close()

if __name__ == "__main__":
    app.run(main=main)