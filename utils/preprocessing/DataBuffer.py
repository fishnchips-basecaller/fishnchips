import numpy as np
from collections import deque

from utils.preprocessing.BacteriaDataLoader import BacteriaDataLoader
from utils.preprocessing.chiron_files.chiron_data_loader import ChironDataLoader

class DataBuffer():

    def __init__(self, filename, bacteria, size=5):
        
        self._position = 0
        self._bacteria = bacteria

        self._loader = ChironDataLoader(filename)
        self._read_ids = self._loader.get_ids()

        self._size = size
        self._signal_windows = []
        self._label_windows = []
        
    def get_windows_in_batch(self, batch_size, window_size, window_stride, min_labels_per_window):
        while (len(self._label_windows) < batch_size):
            self._fetch(window_size, window_stride, min_labels_per_window)
            self._shuffle()

        x = np.array(self._signal_windows[:batch_size])
        y = np.array(self._label_windows[:batch_size])
        
        self._drop(batch_size)
        return x,y

    def get_read_id_idx(self):
        return self._position

    # Used for prediction
    # x_read is the list of windows (consecutive)
    # ref is the reference used to identify the bacteria and perform lcs
    def get_raw_and_split_read(self, window_size, window_stride):
        pos = self._position
        self._position += 1
        read_id = self._read_ids[pos]

        DAC, _, REF = self._loader.get_read(read_id)
        x_read, y_read = self._fetch_read(read_id, window_size, window_stride, min_labels_per_window=0)
        return x_read, y_read, list(REF), DAC, read_id

    def _get_read_ids(self):
        read_ids = self._loader.load_read_ids()
        np.random.shuffle(read_ids)
        return read_ids

    def _drop(self, amount):
        self._signal_windows = self._signal_windows[amount+1:]
        self._label_windows = self._label_windows[amount+1:]

    def _fetch(self, window_size, window_stride, min_labels_per_window):
        
        skips = 0
        found = 0
        while found < self._size:

            read_id_idx = self._position + (skips + found)
            read_id = self._read_ids[read_id_idx]
            
            # is_read_id_valid = self._loader.is_read_id_in_bacteria_lst(read_id, self._bacteria)
            # if is_read_id_valid == False:
            #     skips += 1
            #     continue
            
            read_x, read_y = self._fetch_read(read_id, window_size, window_stride, min_labels_per_window) 

            self._signal_windows.extend(read_x)
            self._label_windows.extend(read_y)
            found += 1
        
        self._position += (skips + found)

    def _shuffle(self):
        x = np.array(self._signal_windows)
        y = np.array(self._label_windows)

        c = np.c_[x.reshape(len(x), -1), y.reshape(len(y), -1)]
        np.random.shuffle(c)
        x_shuffled = c[:, :x.size//len(x)].reshape(x.shape)
        y_shuffled = c[:, x.size//len(x):].reshape(y.shape)
        
        self._signal_windows = x_shuffled.tolist()
        self._label_windows = y_shuffled.tolist()

    def _fetch_read(self, read_id, window_size, window_stride, min_labels_per_window):
        x_read = []
        y_read = []
        print(f"*** fetching id:{read_id}.")       
        DAC, RTS, REF = self._loader.get_read(read_id)
        
        curdacs  = deque( [[x] for x in DAC[RTS[0]:RTS[0]+window_size-window_stride]], window_size )
        curdacts = RTS[0]+ window_size-window_stride
        labels  = deque([])
        labelts = deque([])
        
        while RTS[0] < curdacts:
            labels.append(REF.popleft())
            labelts.append(RTS.popleft())

        while curdacts+window_stride < RTS[-1]-window_size:
            curdacs.extend([[x] for x in DAC[curdacts:curdacts+window_stride]])
            curdacts += window_stride

            while RTS[0] < curdacts:
                labels.append(REF.popleft())
                labelts.append(RTS.popleft())

            while len(labelts) > 0 and labelts[0] < curdacts - window_size:
                labels.popleft()
                labelts.popleft()

            if len(labels) >= min_labels_per_window:
                x_read.append(list(curdacs))
                y_read.append(list(labels))
            
        return (x_read,y_read)