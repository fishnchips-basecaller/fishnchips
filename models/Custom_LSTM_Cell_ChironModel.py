import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Activation, Add, Lambda, Dense, MaxPooling1D, Conv1D, LSTM, GRU, BatchNormalization, RNN, LSTMCell
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras.callbacks import Callback
from functools import reduce
import editdistance
import datetime
import os
import matplotlib.pyplot as plt

from utils.Other import labelBaseMap
from models.RNN_cells.LSTM_Cell_Wrapper import LSTM_Cell_Wrapper
from models.RNN_cells.LSTM_Cell import LSTM_Cell

class Custom_LSTM_Cell_ChironModel():
    
    def __init__(self, max_label_length):
        self.max_label_length = max_label_length
        self.model, self.testfunc = self.make_model()
        
    def predict(self, input_data):
        pred = self.testfunc(input_data)[0]
        cur = [[np.argmax(ts) for ts in p] for p in pred]
        nodup = ["".join(list(map(lambda x: labelBaseMap[x], filter(lambda x: x!=4, reduce(lambda acc, x: acc if acc[-1] == x else acc + [x], c[5:], [4]))))) for c in cur]
        return nodup

    def predict_raw(self, input_data):
        return self.testfunc(input_data)
        
    def make_res_block(self, input, block_number):

        normalized_input = BatchNormalization()(input)

        if block_number==1:
            residual = Conv1D(256, 1, padding="same",name=f"res{block_number}-r")(normalized_input)
        else:
            residual = normalized_input

        conv1 = Conv1D(256, 1, padding="same", activation="relu", use_bias="false", name=f"res{block_number}-c1")(normalized_input)
        conv2 = Conv1D(256, 3, padding="same", activation="relu", use_bias="false", name=f"res{block_number}-c2")(conv1)
        conv3 = Conv1D(256, 1, padding="same", use_bias="false", name=f"res{block_number}-c3")(conv2)

        added = Add(name=f"res{block_number}-add")([residual, conv3])
        return Activation('relu', name=f"res{block_number}-relu")(added)

    def make_bdlstm(self, input, units, block_number):

        cell_fw = LSTM_Cell(units)\
            #with_batch_normalization()
        
        cell_bw = LSTM_Cell(units)\
            #.with_batch_normalization()

        lstm_fw = RNN(cell_fw, return_sequences=True, name=f"blstm{block_number}-fw")(input)
        lstm_bw = RNN(cell_bw, return_sequences=True, go_backwards=True, name=f"blstm{block_number}-bw")(input)
        return Add(name=f"blstm{block_number}-add")([lstm_fw, lstm_bw])

    def make_model(self):
        
        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            # y_pred = y_pred[:, 5:, :]
            return kb.ctc_batch_cost(labels, y_pred, input_length, label_length) 
        
        input_data = Input(name="the_input", shape=(300,1), dtype="float32")

        inner = self.make_res_block(input_data, 1)
        inner = self.make_res_block(inner, 2)
        inner = self.make_res_block(inner, 3)
        inner = self.make_res_block(inner, 4)
        inner = self.make_res_block(inner, 5)
        inner = self.make_bdlstm(inner, 200, 1)
        inner = self.make_bdlstm(inner, 200, 2)
        inner = self.make_bdlstm(inner, 200, 3)

        inner = BatchNormalization()(inner)

        inner = Dense(64, name="dense", activation="relu")(inner)
        inner = Dense(5, name="dense_output")(inner)

        y_pred = Activation("softmax", name="softmax")(inner)

        labels = Input(name='the_labels', shape=(self.max_label_length), dtype='float32')
        input_length = Input(name='input_length', shape=(1), dtype='int64')
        label_length = Input(name='label_length', shape=(1), dtype='int64')

        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out, name="chiron")
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
        
        testfunc = tf.keras.backend.function(input_data, y_pred)
        return model, testfunc
        
    def calculate_loss(self, X, y, testbatchsize=1000):
        editdis = 0
        editdiss = []
        for b in range(0, len(X), testbatchsize):
            predicted = self.predict(X[b:b+testbatchsize])
            mtest_y = ["".join(list(map(lambda x: labelBaseMap[x], ty))) for ty in y[b:b+testbatchsize]]
            for (p,l) in zip(predicted, mtest_y):
                ed = editdistance.eval(p,l)
                editdis += ed
                editdiss.append(ed)
        return (editdis, len(X), editdiss)
    
    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
    