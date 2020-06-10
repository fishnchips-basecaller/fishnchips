import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Add, Lambda, Dense, MaxPooling1D, Conv1D, LSTM, BatchNormalization, Dropout
from tensorflow.keras.backend import ctc_batch_cost, ctc_decode
from functools import reduce
import editdistance
import matplotlib.pyplot as plt

from utils.Other import labelBaseMap

class Gravlax():
    
    def __init__(self, input_length, cnn_filters, lstm_units, rnn_padding, batch_normalization, maxpool_layer, model_name, dropout, use_None_input):
        self.input_length = input_length
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.use_None_input = use_None_input
        self.rnn_output_length = input_length-(2*rnn_padding) if maxpool_layer == 0 else (input_length//2)-(2*rnn_padding)
        self.rnn_padding = rnn_padding
        self.batch_normalization = batch_normalization
        self.maxpool_layer = maxpool_layer
        self.dropout = dropout
        self.name=model_name
        self._model, self.testfunc = self.make_model()
        self.stop_training = False
   
    def fit(self, *args, **kwargs):
        self._model.fit(*args, **kwargs)
    
    def fit_generator(self, *args, **kwargs):
        self._model.fit_generator(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        self._model.load_weights(*args, **kwargs)

    def save_weights(self, *args, **kwargs):
        self._model.save_weights(*args, **kwargs)

    def predict(self, input_data, batchsize = 300):
        results = []
        for i in range(0, len(input_data), batchsize):
            pred = self.testfunc(input_data[i:i+batchsize])
            cur = [[np.argmax(ts) for ts in p] for p in pred]
            nodup = ["".join(list(map(lambda x: labelBaseMap[x], filter(lambda x: x!=4, reduce(lambda acc, x: acc if acc[-1] == x else acc + [x], c[5:], [4]))))) for c in cur]
            results.extend(nodup)
        logs = [1]*len(results) # for compatibility with predict_beam_search
        return results, logs

    def predict_beam_search(self, input_data, batchsize = 300, beam_width=10):
        results = []
        logs = []
        for i in range(0, len(input_data), batchsize):
            pred = self.testfunc(input_data[i:i+batchsize])
            input_lenghts = np.array([self.rnn_output_length]*len(pred))
            greedy = beam_width <= 1
            decoded = ctc_decode(pred, input_lenghts, greedy=greedy, beam_width=beam_width, top_paths=1)

            # transform the actual tensor output of decoded into a string with the labels
            # decoded[0] contains one list with the outputs -> decoded[0][0] is the list of outputs
            # decoded[1] is the list of log likelihoods
            transformed_to_bases = []
            for d in decoded[0][0]:
                l = list(filter(lambda x: x>=0, np.array(d)))
                tr = "".join([labelBaseMap[x] for x in l])
                results.append(tr)
            logs.extend(np.array(decoded[1]))
        return results, logs

    def predict_raw(self, input_data):
        return self.testfunc(input_data)
        
    def make_res_block(self, upper, block):

        inner = upper
        if(self.batch_normalization):
            inner = BatchNormalization()(upper)

        if block==1:
            res = Conv1D(self.cnn_filters, 1,
                padding="same",
                name=f"res{block}-r")(inner)
        else:
            res = inner

        inner = Conv1D(self.cnn_filters, 1,
                      padding="same",
                      activation="relu",
                      name=f"res{block}-c1")(inner)
        inner = Conv1D(self.cnn_filters, 3,
                      padding="same",
                      activation="relu",
                      name=f"res{block}-c2")(inner)
        inner = Conv1D(self.cnn_filters, 1,
                      padding="same",
                      name=f"res{block}-c3")(inner)

        added = Add(name=f"res{block}-add")([res, inner])
        act = Activation('relu', name=f"res{block}-relu")(added)
        if self.dropout:
            return Dropout(0.1, name=f"res{block}-dropout")(act)
        else:
            return act

    def make_bdlstm(self, upper, block):
        
        inner = upper
        if(self.batch_normalization):
            inner = BatchNormalization()(upper)

        lstm_1a = LSTM(self.lstm_units, return_sequences=True, name=f"blstm{block}-fwd")(inner)
        lstm_1b = LSTM(self.lstm_units, return_sequences=True, go_backwards=True, name=f"blstm{block}-rev")(inner)
        return Add(name=f"blstm{block}-add")([lstm_1a, lstm_1b])

    def make_model(self):
        
        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            if self.rnn_padding > 0:
                y_pred = y_pred[:, self.rnn_padding:-self.rnn_padding, :]
            return ctc_batch_cost(labels, y_pred, input_length, label_length) 
        
        if self.use_None_input:
            input_data = Input(name="the_input", shape=(None,1), dtype="float32")
        else:
            input_data = Input(name="the_input", shape=(self.input_length,1), dtype="float32")

        if self.dropout:
            inner = Dropout(0.2, name="input-dropout")(input_data)
        else:
            inner = input_data

        for res_idx in range(1,6):
            inner = self.make_res_block(inner, res_idx)
            if self.maxpool_layer == res_idx:
                inner = MaxPooling1D(pool_size=2, name="max_pool_1D")(inner)


        inner = self.make_bdlstm(inner, 1)
        inner = self.make_bdlstm(inner, 2)
        inner = self.make_bdlstm(inner, 3)

        if(self.batch_normalization):
            inner = BatchNormalization()(inner)
        
        if self.dropout:
            inner = Dropout(0.3, name="dense_dropout")(inner)
        inner = Dense(64, name="dense", activation="relu")(inner)
        inner = Dense(5, name="dense_output")(inner)

        y_pred = Activation("softmax", name="softmax")(inner)

        labels = Input(name='the_labels', shape=(self.rnn_output_length), dtype='float32')
        input_length = Input(name='input_length', shape=(1), dtype='int64')
        label_length = Input(name='label_length', shape=(1), dtype='int64')

        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out, name="gravlax")
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
        
        testfunc = tf.keras.backend.function(input_data, y_pred)
        return model, testfunc
