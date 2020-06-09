import math
from collections import deque
import random

from utils.DataLoader import DataLoader

class DataPreperVarLen():
    def __init__(self, filename=None, validation_split=0.1, test_split=0.1, window_sizes=[300], window_skip=5):

        self._validation_split = validation_split
        self._test_split = test_split
        self._train_split = self._get_train_split()
        
        self._loader = DataLoader(filename)
        self._window_sizes = window_sizes
        self._window_skip = window_skip

        self._read_ids = []
        self._train_read_ids = []
        self._test_read_ids = []
        self._validate_read_ids = []
        self._set_read_ids()
        
        self._set_max_label_size()

    def get_read_ids(self):
        return self._read_ids

    def get_no_train_example(self):
        return len(self._train_read_ids)

    def get_no_validation_example(self):
        return len(self._validate_read_ids)

    def get_no_test_examples(self):
        return len(self._test_read_ids)

    def get_max_input_size(self):
        return max(self._window_sizes)

    def get_max_label_size(self):
        return self._max_label_length

    def _get_read_ids_slice(self, start_index, end_index):
        if(start_index < end_index):
            return self._read_ids[start_index:end_index]
        return []
    
    def _get_train_split(self):
        if(self._validation_split + self._test_split > 1):
            raise AttributeError("Data preper: Invalid train, test and validation split. Split cannot exceed 1.")
        return 1 - self._validation_split - self._test_split

    def _get_window_size(self):
        return random.choice(self._window_sizes)

    def _set_max_label_size(self, with_heuristics = True):
        if(with_heuristics):
            self._max_label_length = 0.2 * self.get_max_input_size()
        else:
            max_label_length = -1
            for i,read_id in enumerate(self._read_ids):
                label_length = self._loader.load_read_label_length(read_id)
                if(label_length > max_label_length):
                    max_label_length = label_length
            self._max_label_length = max_label_length

    def _set_read_ids(self):
        self._read_ids = self._loader.load_read_ids()
        

    def split_read_ids(self, epoch):

        no_ids = len(self._read_ids)
        no_splits = math.floor(self._train_split / self._validation_split) + 1

        split_number = round(epoch % no_splits, 2)

        train_split_1 = self._train_split - split_number * self._validation_split
        train_split_1_end_index = math.floor(no_ids * train_split_1)
        train_read_ids_1 = self._get_read_ids_slice(0, train_split_1_end_index)
        
        validation_split_start_index = train_split_1_end_index + 1
        validation_split_end_index = math.floor(no_ids * (train_split_1 + self._validation_split))
        validation_read_ids = self._get_read_ids_slice(validation_split_start_index, validation_split_end_index)

        train_split_2 = self._train_split - train_split_1
        train_split_2_start_index = validation_split_end_index + 1
        train_split_2_end_index = math.floor(no_ids * (train_split_1 + self._validation_split + train_split_2))
        train_read_ids_2 = self._get_read_ids_slice(train_split_2_start_index, train_split_2_end_index)

        test_split_start_index = train_split_2_end_index + 1
        test_read_ids = self._get_read_ids_slice(test_split_start_index, no_ids)
        
        self._train_read_ids = train_read_ids_1 + train_read_ids_2
        self._validate_read_ids = validation_read_ids
        self._test_read_ids = test_read_ids

    def process_read(self, read_id, window_size, window_skip):
        x = []
        y = []       
        DAC, RTS, REF = self._loader.load_read(read_id)

        curdacs  = deque( [[x] for x in DAC[RTS[0]:RTS[0]+window_size-window_skip]], window_size )
        curdacts = RTS[0]+ window_size-window_skip
        labels  = deque([])
        labelts = deque([])
        
        while RTS[0] < curdacts:
            labels.append(REF.popleft())
            labelts.append(RTS.popleft())

        while curdacts+window_skip < RTS[-1]-window_size:
            curdacs.extend([[x] for x in DAC[curdacts:curdacts+window_skip]])
            curdacts += window_skip

            while RTS[0] < curdacts:
                labels.append(REF.popleft())
                labelts.append(RTS.popleft())

            while len(labelts) > 0 and labelts[0] < curdacts - window_size:
                labels.popleft()
                labelts.popleft()

            x.append(list(curdacs))
            y.append(list(labels))
            
        return (x,y) 

    def get_training_example(self, read_id):
        
        print(f"Processing read id # {read_id}...")
        window_size = self._get_window_size()
        return self.process_read(read_id, window_size, self._window_skip)
    

    
