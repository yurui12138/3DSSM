# -*- coding: utf-8 -*-
import pickle
import sys
import time

import nni
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.backend import stack
from keras.engine import Layer
from keras.layers import Concatenate, Add, Flatten, Multiply, Conv3D, GRU
import keras
from keras.utils import plot_model

import data_helper
import data_helper_word

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

KTF.set_session(sess)
from keras import backend as K, backend
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.layers import Embedding, Input, Bidirectional, Lambda, LSTM, \
    Reshape, Dense, Activation, concatenate, BatchNormalization, Conv2D, MaxPooling3D
from keras.models import Model
from keras.optimizers import RMSprop


input_dim = data_helper.MAX_SEQUENCE_LENGTH
emb_dim = data_helper.EMB_DIM
model_path = './model/siameselstm.hdf5'
tensorboard_path = './model/ensembling'


def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def precision(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2
    return precision


def recall(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    recall = c1 / c3
    return recall


class Attention(Layer):
    def __init__(self,**kwargs):

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)

        self.Wd = self.add_weight(name='Wd',
                                  shape=(input_shape[0][2], 1),
                                  initializer='uniform',
                                  trainable=True)
        self.vd = self.add_weight(name='vd',
                                  shape=(input_shape[0][1], 1),
                                  initializer='uniform',
                                  trainable=True)


        super(Attention, self).build(input_shape) 

    def call(self, x):
        q_sentence_output, p_sentence_output = x

        q_p_dot = tf.expand_dims(q_sentence_output, axis=1) * tf.expand_dims(p_sentence_output, axis=2)
        sd = tf.multiply(tf.tanh(K.dot(q_p_dot, self.Wd)), self.vd)  
        sd = tf.squeeze(sd, axis=-1)
        ad = tf.nn.softmax(sd)  
        qd = K.batch_dot(ad, q_sentence_output) 

        return [qd]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return [input_shape[0]]

class self_Attention(Layer):
    def __init__(self,**kwargs):

        super(self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.Wd = self.add_weight(name='Wd',
                                  shape=(input_shape[2], 1),
                                  initializer='uniform',
                                  trainable=True)
        self.vd = self.add_weight(name='vd',
                                  shape=(input_shape[1], 1),
                                  initializer='uniform',
                                  trainable=True)


        super(self_Attention, self).build(input_shape)  

    def call(self, x):
        sentence_output= x

        sd = tf.multiply(tf.tanh(K.dot(sentence_output, self.Wd)), self.vd)  
        sd = tf.squeeze(sd, axis=-1)
        ad = tf.nn.softmax(sd)  
        qd = K.batch_dot(ad, sentence_output)

        return [qd]

    def compute_output_shape(self, input_shape):
        # assert isinstance(input_shape, list)
        return (input_shape[0], input_shape[2])

embedding_matrix = data_helper.load_pickle('./embedding_matrix.pkl')
embedding_layer = Embedding(embedding_matrix.shape[0],
                            emb_dim,
                            weights=[embedding_matrix],
                            input_length=input_dim,
                            trainable=False)    

embedding_matrix_word = data_helper_word.load_pickle('./embedding_matrix_word.pkl')
embedding_layer_word = Embedding(embedding_matrix_word.shape[0],
                            emb_dim,
                            weights=[embedding_matrix_word],
                            input_length=input_dim,
                            trainable=False)     

def encoding_moudle(input_shape,params):
    sentence_char = Input(shape=input_shape, dtype='int32')
    sentence_word = Input(shape=input_shape, dtype='int32')

    sentence_embed = embedding_layer(sentence_char)
    sentence_embed_word = embedding_layer_word(sentence_word)

    encoded_result = []
    sentence_next = sentence_embed
    sentence_next_word = sentence_embed_word
    for i in range(1,params["depth"]+1):
        encoder = Bidirectional(LSTM(params["lstm_dim"],
                                    return_sequences=True,
                                    dropout=params["dropout"]),
                                    merge_mode='sum',
                                    name="BiLSTM"+str(i))
        sentence_next = encoder(sentence_next)
        sentence_next_word = encoder(sentence_next_word)
        sentence_bilstm = Reshape((30, 1, params["lstm_dim"], 1))(sentence_next)
        sentence_bilstm_word = Reshape((30, 1, params["lstm_dim"], 1))(sentence_next_word)


        feature_map = Concatenate(axis=2)([sentence_bilstm, sentence_bilstm_word])
        encoded_result.append(feature_map)

    feature_map = encoded_result[0]
    for i in range(len(encoded_result)-1):
        feature_map = Concatenate(axis=4)([feature_map, encoded_result[i+1]])

    encode_3DCNN = Conv3D(filters=params["conv3d_dim"], kernel_size=(params["conv3d_kernel_1"], 2, params["conv3d_kernel_3"]),
                  strides=(params["conv3d_stride_1"], 1, params["conv3d_stride_3"]),
                  padding='valid', data_format='channels_last',
                  activation='relu')(feature_map)

    encoding_moudle = Model([sentence_char, sentence_word], encode_3DCNN, name='encoding_moudle')
    encoding_moudle.summary()
    plot_model(encoding_moudle, to_file='encoding_moudle.png', show_shapes=True)
    return encoding_moudle

def siamese_model(params):
    input_shape = (input_dim,)

    encode_moudel = encoding_moudle(input_shape, params)

    q1_input_char = Input(shape=input_shape, dtype='int32', name='sequence1_char')
    q1_input_word = Input(shape=input_shape, dtype='int32', name='sequence1_word')
    q1_feature = encode_moudel([q1_input_char,q1_input_word])

    q2_input_char = Input(shape=input_shape, dtype='int32', name='sequence2_char')
    q2_input_word = Input(shape=input_shape, dtype='int32', name='sequence2_word')
    q2_feature = encode_moudel([q2_input_char,q2_input_word])

    similarity = Concatenate(axis=4)([q1_feature, q2_feature])

    similarity = Conv3D(filters=params["sim_conv3d_dim"], kernel_size=(params["sim_conv3d_kernel_1"], 1, params["sim_conv3d_kernel_3"]),
                  strides=(params["sim_conv3d_stride_1"], 1, params["sim_conv3d_stride_3"]),
                  padding='valid', data_format='channels_last',
                  activation='relu')(similarity)

    similarity = MaxPooling3D(pool_size=(3,1,5), strides=[2,1,3], padding="valid", data_format="channels_last")(similarity)


    _dim1 = backend.int_shape(similarity)[1]
    _dim2 = backend.int_shape(similarity)[2]
    _dim3 = backend.int_shape(similarity)[3]
    _dim4 = backend.int_shape(similarity)[4]


    similarity = Reshape((_dim1*_dim3, _dim4))(similarity)

    similarity = self_Attention()(similarity)

    similarity = Dense(1)(similarity)
    similarity = BatchNormalization()(similarity)
    similarity = Activation('sigmoid')(similarity)

    model = Model([q1_input_char, q1_input_word, q2_input_char, q2_input_word], [similarity])
    # summarize layers
    model.summary()
    plot_model(model, to_file='framework.png', show_shapes=True)

    op = RMSprop(lr=0.0015)

    from keras.utils import multi_gpu_model
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss="binary_crossentropy", optimizer=op, metrics=['accuracy', precision, recall, f1_score])

    return parallel_model

def train(params):
    data = data_helper.load_pickle('./model_data.pkl')

    train_q1 = data['train_q1']
    train_q2 = data['train_q2']
    train_y = data['train_label']

    dev_q1 = data['dev_q1']
    dev_q2 = data['dev_q2']
    dev_y = data['dev_label']

    test_q1 = data['test_q1']
    test_q2 = data['test_q2']
    test_y = data['test_label']

    data_word = data_helper_word.load_pickle('./model_data_word.pkl')

    train_q1_word = data_word['train_q1']
    train_q2_word = data_word['train_q2']

    dev_q1_word = data_word['dev_q1']
    dev_q2_word = data_word['dev_q2']

    test_q1_word = data_word['test_q1']
    test_q2_word = data_word['test_q2']

    model = siamese_model(params)
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max', period=1)
    tensorboard = TensorBoard(log_dir=tensorboard_path)
    earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max',restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=3, mode='max')
    callbackslist = [checkpoint, tensorboard, earlystopping, reduce_lr]

    model.fit([train_q1, train_q1_word, train_q2, train_q2_word], train_y,
              batch_size=512,
              epochs=200,
              verbose=2,
              validation_data=([dev_q1, dev_q1_word, dev_q2, dev_q2_word], dev_y),
              callbacks=callbackslist)

    loss, accuracy, precision, recall, f1_score = model.evaluate(
        [test_q1, test_q1_word, test_q2, test_q2_word], test_y, verbose=1, batch_size=256)


    print("Test best model =loss: %.4f, accuracy:%.4f,precision:%.4f,recall:%.4f,f1_score:%.4f" % (
    loss, accuracy, precision, recall, f1_score))
    # nni.report_final_result(accuracy)

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == '__main__':

    params63 = {"lstm_dim": 300,
                "conv3d_dim": 16,
                "conv3d_kernel_1": 2, "conv3d_stride_1": 1,
                "conv3d_kernel_3": 5, "conv3d_stride_3": 1,
                "sim_conv3d_dim": 32,
                "sim_conv3d_kernel_1": 5, "sim_conv3d_stride_1": 1,
                "sim_conv3d_kernel_3": 2, "sim_conv3d_stride_3": 1,
                "dropout": 0.05, "depth":4}

    train(params63)
