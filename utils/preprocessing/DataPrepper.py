import math
from collections import deque
import numpy as np

from utils.preprocessing.DataLoader import DataLoader

class DataPrepper():
    
    def __init__(self, training, validation_split=0.1, test_split=0.1):    
        self._loader = DataLoader(training)
        self._validation_split = validation_split
        self._test_split = test_split
        self._train_split = self._get_train_split()

        self._read_ids = []
        self._train_read_ids = []
        self._test_read_ids = []
        self._validate_read_ids = []

        self._set_read_ids()
        self._split_read_ids()

    def get_train_read_ids(self):
        return self._train_read_ids

    def get_validation_read_ids(self):
        return self._validate_read_ids

    def get_test_read_ids(self):
        return self._test_read_ids

    def get_all_read_ids(self):
        return self._read_ids

    def _get_train_split(self):
        if(self._validation_split + self._test_split > 1):
            raise AttributeError("Data preper: Invalid train, test and validation split. Split cannot exceed 1.")
        return 1 - self._validation_split - self._test_split

    def _set_read_ids(self):
        read_ids = self._loader.load_read_ids()
        self._read_ids = np.random.shuffle(read_ids)
        self._read_ids = read_ids
    
    def _split_read_ids(self):
        no_ids = len(self._read_ids)
        train_split_end_index = math.floor(no_ids * self._train_split)
        validate_split_start_index = train_split_end_index + 1
        validate_split_end_index = math.floor(no_ids * (self._train_split + self._validation_split))
        test_split_start_index = validate_split_end_index + 1

        self._train_read_ids = self._read_ids[:train_split_end_index]
        self._validate_read_ids = self._read_ids[validate_split_start_index:validate_split_end_index]
        self._test_read_ids = self._read_ids[test_split_start_index:]     