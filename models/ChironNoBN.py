import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Activation, Add, Lambda, Dense, MaxPooling1D, Conv1D, LSTM, GRU, BatchNormalization
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras.callbacks import Callback
from functools import reduce
from Utils import labelBaseMap
import editdistance
import datetime
import os
import matplotlib.pyplot as plt

class Chiron():
    
    def __init__(self, max_label_length, input_raw_length=300):
        self.max_label_length = max_label_length
        self.model, self.testfunc = self.make_model(input_raw_length)
        
    def predict(self, input_data):
        pred = self.testfunc(input_data)
        cur = [[np.argmax(ts) for ts in p] for p in pred]
        nodup = ["".join(list(map(lambda x: labelBaseMap[x], filter(lambda x: x!=4, reduce(lambda acc, x: acc if acc[-1] == x else acc + [x], c[5:], [4]))))) for c in cur]
        return nodup

    def predict_raw(self, input_data):
        return self.testfunc(input_data)
        
    def make_res_block(self, upper, block):

        # inner = BatchNormalization()(upper)
        inner = upper

        if block==1:
            res = Conv1D(256, 1,
                padding="same",
                name=f"res{block}-r")(inner)
        else:
            res = inner

        inner = Conv1D(256, 1,
                      padding="same",
                      activation="relu",
                      use_bias="false",
                      name=f"res{block}-c1")(inner)
        inner = Conv1D(256, 3,
                      padding="same",
                      activation="relu",
                      use_bias="false",
                      name=f"res{block}-c2")(inner)
        inner = Conv1D(256, 1,
                      padding="same",
                      use_bias="false",
                      name=f"res{block}-c3")(inner)

        added = Add(name=f"res{block}-add")([res, inner])
        return Activation('relu', name=f"res{block}-relu")(added)

    def make_bdlstm(self, upper, block):
        # inner = BatchNormalization()(upper)
        inner = upper

        lstm_1a = LSTM(200, return_sequences=True, name=f"blstm{block}-fwd")(inner)
        lstm_1b = LSTM(200, return_sequences=True, go_backwards=True, name=f"blstm{block}-rev")(inner)
        return Add(name=f"blstm{block}-add")([lstm_1a, lstm_1b])

    def make_model(self, input_raw_length):
        
        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            # y_pred = y_pred[:, 5:, :]
            return kb.ctc_batch_cost(labels, y_pred, input_length, label_length) 
        
        input_data = Input(name="the_input", shape=(input_raw_length,1), dtype="float32")

        inner = self.make_res_block(input_data, 1)
        inner = self.make_res_block(inner, 2)
        inner = self.make_res_block(inner, 3)
        inner = self.make_res_block(inner, 4)
        inner = self.make_res_block(inner, 5)
        inner = self.make_bdlstm(inner, 1)
        inner = self.make_bdlstm(inner, 2)
        inner = self.make_bdlstm(inner, 3)

        # inner = BatchNormalization()(inner)

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
        
    

    
    
class SaveCB(Callback):
    def __init__(self, model_output_dir, image_output_dir, chiron, prepper):
        self.model_output_dir=model_output_dir
        self.image_output_dir=image_output_dir
        self.chiron = chiron
        self.prepper = prepper
        self.best_dist = None
        self.Xforimg = None
        self.testvalid = [[],[],[]]
        self.start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        if not os.path.exists(self.model_output_dir):
            os.makedirs(self.model_output_dir)
        if not os.path.exists(self.image_output_dir):
            os.makedirs(self.image_output_dir)
        
    def save_anim_pic(self, epoch):
        fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(30,10))
        ax.set_ylim(top=1)
        ax.set_ylim(bottom=0)
        prediction = self.chiron.predict_raw(self.Xforimg)[0]
        transposed = list(map(list, zip(*prediction)))
        for i in range(len(transposed)):
            ax.plot(transposed[i], label=labelBaseMap[i])
        ax.plot(self.Xforimg[0], "k", label="raw")
        ax.legend()
        fig.savefig(os.path.join(self.image_output_dir, f'{self.start_time}-{epoch:05d}.png'))
        plt.close(fig)
    

    def on_epoch_end(self, epoch, logs={}):
        test_X, test_y = next(self.prepper.test_gen())
        train_X, train_y = self.prepper.last_train_gen_data[0]['the_input'], self.prepper.last_train_gen_data[0]['unpadded_labels']
        if self.Xforimg is None:
            self.Xforimg = test_X[0:1]
        self.save_anim_pic(epoch)

        totloss, n, _ = self.chiron.calculate_loss(train_X, train_y)
        testloss = totloss/n
        print(f"\nAverage test edit distance is: {testloss}")
        totloss, n, _ = self.chiron.calculate_loss(test_X, test_y)
        valloss = totloss/n
        print(f"\nAverage validation edit distance is: {valloss}")
        self.testvalid[0].append(testloss)
        self.testvalid[1].append(valloss)
        self.testvalid[2].append(int(datetime.datetime.now().timestamp()))
        np.save(os.path.join(self.model_output_dir, self.start_time), np.array(self.testvalid))
        
        if self.best_dist is None or valloss < self.best_dist:
            self.best_dist = valloss
            self.model.save_weights(os.path.join(self.model_output_dir, f'{self.start_time}_e{epoch:05d}_dis{round(valloss*100)}.h5'))
        elif epoch%20==0: # so it doesn't reset best_dist
            self.model.save_weights(os.path.join(self.model_output_dir, f'{self.start_time}_e{epoch:05d}_dis{round(valloss*100)}.h5'))

    