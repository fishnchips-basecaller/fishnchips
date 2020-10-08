import sys
import numpy as np

from os import listdir
from collections import deque
from utils.Other import attentionLabelBaseReverseMap

class ChironDataLoader:

    def __init__(self, data_dir): 
        self._construct_file_dict(data_dir) #"./data/train")
        self._construct_key_lst()
    
    def get_ids(self):
        return self._ids

    def _construct_file_dict(self, dir):   
        file_dict = {}

        for filename in listdir(dir):
            filepath = f"{dir}/{filename}"
            name, extension = filename.split(".")

            if name not in file_dict:
                file_dict[name] = {}      
            file_dict[name][extension] = filepath
        self._file_dict = file_dict

    def _construct_key_lst(self):
        arr = self._file_dict.keys()
        arr = np.array(list(arr))
        np.random.shuffle(arr)
        self._ids = arr

    def _process_label_str(self, label_str):
        ref = []
        rts = []
        
        label = label_str.split('\n')[:-1]
        for base_obj in label:
            split = base_obj.split(' ')
            rts.append(int(split[0]))
            ref.append(attentionLabelBaseReverseMap[split[2]])

        rts.append(int(label[-1].split(' ')[1]))
        return deque(ref), deque(rts)

    def _load_file(self, filename):
        with open(filename, 'r') as f:
            return f.read()

    def _normilize_signal(self, signal):
        signal = np.array(signal).astype(np.int32)
        return (signal - np.mean(signal)/np.std(signal))

    def get_read(self, idx):
        assert len(self._file_dict[idx]) == 2, "Failed to construct data. Signal / label is missing or the files are not structured correctly."

        data = []
        label_str = self._load_file(self._file_dict[idx]["label"])
        signal_str = self._load_file(self._file_dict[idx]["signal"])

        ref, rts = self._process_label_str(label_str)
        dac = list(map(int, signal_str.split(' ')))
        dac = self._normilize_signal(dac)
        return dac, rts, ref