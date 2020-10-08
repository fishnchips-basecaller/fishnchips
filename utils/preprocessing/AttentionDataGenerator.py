import numpy as np
import json
import os
import sys

from utils.preprocessing.DataGenerator import DataGenerator
from utils.Other import attentionLabelBaseMap

class AttentionDataGenerator(DataGenerator):
    def __init__(self, filename, bacteria, batch_size, stride, pe_encoder_max_length, pe_decoder_max_length):
        super().__init__(filename, bacteria, batch_size, stride, input_length=pe_encoder_max_length, reads_count=5, use_maxpool=False, rnn_pad_size=0)
        
        self._pe_decoder_max_length = pe_decoder_max_length
        self._batch_count = 0

        # self._set_umi_to_bact_dict()

    def _set_umi_to_bact_dict(self):
        path = "./temps/uids.json"
        if os.path.isfile(path) == False:
            print("UMI to bacteria mappings have not been generated. Please run 'create_umi_to_bacteria_dict.py' script.")
            sys.exit()

        with open(path) as f:
            self._umibactdict = json.load(f)

    def get_batch(self, label_as_bases=False):
        while True:
            self._batch_count += 1
            
            x, y_orig = self._buffer.get_windows_in_batch(self.batch_size, self.input_length, self.stride, min_labels_per_window=1)  
            y = self._to_target_language(y_orig, label_as_bases)
            yield (x,y)
    
    def get_batches(self, number_of_batches, label_as_bases=False):
        while True:
            batches = []
            for _ in range(number_of_batches):
                x,y = next(self.get_batch(), label_as_bases)
                batches.append([x,y])

            #batches = np.array(batches)
            yield batches

    def get_window_batch(self, label_as_bases=False):
        while True:
            self._batch_count += 1
            x_windows, y_orig_windows, ref, raw, read_id = self._buffer.get_raw_and_split_read(
                self.input_length,
                self.stride
            )

            x_windows = np.array(x_windows)
            y_windows = self._to_target_language(y_orig_windows, label_as_bases)
            yield x_windows, y_windows, ref, raw, read_id

    def _to_target_language(self, y_orig, as_bases):
        y_new = []
        for y in y_orig:
            y = [t+1 for t in y] # since 0 is a base
            y.insert(0, 5) # add 5 as start token
            y.append(6) # add 6 as end token
            y.extend([0]*(self._pe_decoder_max_length-len(y))) # pad with zeros to pe_decoder_max_length
            if as_bases:
                y = "".join([attentionLabelBaseMap[base_token] if (base_token in [1,2,3,4]) else "" for base_token in y])
            
            y_new.append(y)
        return np.array(y_new)



