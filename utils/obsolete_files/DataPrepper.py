from collections import deque
from tensorflow.keras.utils import Sequence
import h5py
import numpy as np


class PrepData(Sequence):
    
    def __init__(self, filename, RNN_LEN, train_validate_split=0.8, min_labels=0):
        self.filename = filename
        self.RNN_LEN = RNN_LEN
        self.train_validate_split=train_validate_split
        self.min_labels=min_labels
        self.pos = 0
        self.test_gen_data = ([],[])
        self.last_train_gen_data = ({},{})
        self.max_label_len = 50
        self.last_orig_ref = []
        self.raw = []
        with h5py.File(filename, 'r') as h5file:
            self.readIDs = list(h5file['Reads'].keys())
            
    def get_len(self):
        return len(self.readIDs)
    
    def get_max_label_len(self):
        return self.max_label_len
        
    def normalise(self, dac):
        dmin = min(dac)
        dmax = max(dac)
        return [(d-dmin)/(dmax-dmin) for d in dac]

    def get_whole_read(self):
        while self.pos < len(self.readIDs):
            readID = self.readIDs[self.pos]
            with h5py.File(self.filename, 'r') as h5file:
                DAC = list(self.normalise(h5file['Reads'][readID]['Dacs'][()]))
                REF = h5file['Reads'][readID]['Reference'][()]

            self.pos += 1
            yield DAC, REF
    
    def processRead(self, readID):
        train_X = []
        train_y = []
        test_X  = []
        test_y  = []
        with h5py.File(self.filename, 'r') as h5file:
            DAC = list(self.normalise(h5file['Reads'][readID]['Dacs'][()]))
            RTS = deque(list(h5file['Reads'][readID]['Ref_to_signal'][()]))
            self.last_orig_ref = h5file['Reads'][readID]['Reference'][()]
            REF = deque(h5file['Reads'][readID]['Reference'][()])
            self.raw = DAC
            
        train_validate_split = round(len(REF)*(1-self.train_validate_split))
        curdacs  = deque( [[x] for x in DAC[RTS[0]:RTS[0]+self.RNN_LEN-5]], self.RNN_LEN )
        curdacts = RTS[0]+self.RNN_LEN-5
        labels  = deque([])
        labelts = deque([])

        while RTS[0] < curdacts:
            labels.append(REF.popleft())
            labelts.append(RTS.popleft())


        while curdacts+5 < RTS[-1]-self.RNN_LEN:
            curdacs.extend([[x] for x in DAC[curdacts:curdacts+5]])
            curdacts += 5
            
            while RTS[0] < curdacts:
                labels.append(REF.popleft())
                labelts.append(RTS.popleft())
                
            while len(labelts) > 0 and labelts[0] < curdacts - self.RNN_LEN:
                labels.popleft()
                labelts.popleft()

            if len(labels) > self.min_labels:
                if len(RTS) > train_validate_split:
                    train_X.append(list(curdacs))
                    train_y.append(list(labels))
                else:
                    test_X.append(list(curdacs))
                    test_y.append(list(labels))

        return train_X, train_y, test_X, test_y
    
    
    def train_gen(self, ignore_boundary_count=0, full=True):
        while self.pos < len(self.readIDs):
            print(f"Processing {self.pos}")
            train_X, train_y, test_X, test_y = self.processRead(self.readIDs[self.pos])
            self.pos += 1
            
            train_X = np.array(train_X) if full else np.array(train_X[:100])
            train_y = np.array(train_y) if full else np.array(train_y[:100])
            test_X  = np.array(test_X) if full else np.array(test_X[:100])
            test_y  = np.array(test_y) if full else np.array(test_y[:100])
            self.test_gen_data = (test_X, test_y)
            
            train_X_lens = np.array([[self.RNN_LEN-ignore_boundary_count] for x in train_X], dtype="float32")
            train_y_lens = np.array([[len(x)] for x in train_y], dtype="float32")
            
            # sometimes there are sequences that exceed max_label_len
            # catch them, remove them, and print message
            maxlen = max([len(r) for r in train_y])
            prevlen = len(train_y)
            if maxlen > self.max_label_len:
                print(f"Caution: longer labels than max len, saw {maxlen} > {self.max_label_len}.")
                train_y = [r for r in train_y if len(r) <= self.max_label_len]
                print(f"Kept {len(train_y)} out of {prevlen}")

            train_y_padded = np.array([r + [5]*(self.max_label_len-len(r)) for r in train_y], dtype='float32')
            X = {'the_input': train_X,
                      'the_labels': train_y_padded,
                      'input_length': train_X_lens,
                      'label_length': train_y_lens,
                      'unpadded_labels' : train_y
                      }
            y = {'ctc': np.zeros([len(train_X)])}
            self.last_train_gen_data = (X, y)
            yield (X, y)
        
    def test_gen(self):
        while True:
            tgd, self.test_gen_data = self.test_gen_data, ([],[])
            yield tgd
            
            
    def __len__(self):
        return len(self.readIDs)

    def __getitem__(self, idx):
        return next(self.train_gen())