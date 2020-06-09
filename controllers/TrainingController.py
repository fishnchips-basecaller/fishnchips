import tensorflow as tf
import time
import numpy as np
import os
import sys

from controllers.ValidationController import ValidationController
from utils.process_utils import get_generator
from models.Attention.CustomSchedule import CustomSchedule
from models.Attention.attention_utils import create_combined_mask
from utils.attention_evaluation_utils import build

class TrainingController():
    def __init__(self, model_config, train_config, validation_config, model, model_filepath):
        self._epochs = train_config['epochs']
        self._patience = train_config['patience']
        self._warmup = train_config['warmup']
        self._batches = train_config['batches']
        self._batch_size = train_config['batch_size']
        self._stride = train_config['stride']

        self._generator = get_generator(model_config, train_config, kind="training")
        self._validation_controller = ValidationController(model_config, validation_config)
        self._model = model
        self._model_filepath = model_filepath
        self._train_loss = tf.keras.metrics.Mean(name='train_loss')
        self._train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self._loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none') 
        self._val_algo = validation_config['algorithm']
        
        learning_rate = CustomSchedule(model_config['d_model']*train_config['lr_mult'])
        self._optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    def retrain_weights(self):

        if (self._epochs == 0 and os.path.exists(f"{self._model_filepath}.h5") == False):	
            print("*** attemt to skip training, but weights are not provided, exiting...")	
            sys.exit()

        if os.path.exists(f"{self._model_filepath}.h5"):
            answer = input(f"*** a trained model already exist, are you sure you want to retrain it? [y/N]:")
            if answer not in "Yy" or answer is "":
                return False
        return True 

    def _load_model_from_file(self):
        print("*** loading trained model from a file and skipping training...")
        build(self._model)
        self._model.load_weights(f"{self._model_filepath}.h5")
        return self._model

    def train(self):

        if self.retrain_weights() == False:
            return self._load_model_from_file()

        print("*** training...")

        waited = 0
        old_acc = 1e10
        accs = []
        weights = None
        for epoch in range(self._epochs):
            if epoch < self._warmup:
                waited = 0
            
            start = time.time()
            self._train_loss.reset_states()
            self._train_accuracy.reset_states()

            batches = next(self._generator.get_batches(self._batches))
            for (batch, (inp, tar)) in enumerate(batches):  
                inp = tf.constant(inp, dtype=tf.float32)
                tar = tf.constant(tar, dtype=tf.int32)
                self.train_step(inp, tar)
                print (f'Epoch {epoch + 1} Batch {batch} Loss {self._train_loss.result():.4f} Accuracy {self._train_accuracy.result():.4f}', end="\r")
            print()
            

            val_loss = self._validation_controller.validate(self._model)
            accs.append([self._train_loss.result(), self._train_accuracy.result(), val_loss, time.time()])
            np.save(f"{self._model_filepath}.npy", np.array(accs)) 

            loss = self._train_loss.result()
            acc = self._train_accuracy.result()
            print (f'Epoch {epoch + 1} Loss {loss:.4f} Accuracy {acc:.4f}, valloss {val_loss}, took {time.time() - start} secs')

            if val_loss < old_acc:
                waited = 0
                old_acc = val_loss
                print("*** saving model weights")
                self._model.save_weights(f"{self._model_filepath}.h5")
                weights = self._model.get_weights()
            else:
                waited += 1
                if waited > self._patience:
                    print("Out of patience, exiting...")
                    break

        self._model.set_weights(weights)
        return self._model  
    
    # train_step_signature = [
    #     tf.TensorSpec(shape=(batch_size, encoder_max_length, 1), dtype=tf.float32),
    #     tf.TensorSpec(shape=(batch_size, decoder_max_length), dtype=tf.int32)
    # ]
    @tf.function
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        combined_mask = create_combined_mask(tar_inp)
        
        with tf.GradientTape() as tape:
            predictions, _ = self._model(inp, tar_inp, True, combined_mask)   
            loss = self._model.get_loss(tar_real, predictions, self._loss_object)

        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

        self._train_loss(loss)
        self._train_accuracy(tar_real, predictions)