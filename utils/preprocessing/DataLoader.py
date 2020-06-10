from collections import deque
import h5py
import numpy as np

class DataLoader():
    def __init__(self, training, filename=None):
        #self._set_filename(filename, training)
        self._filename = filename

    def _normalize_signal(self, signal):
        signal = np.array(signal)
        mean = np.mean(signal)
        standard_dev = np.std(signal)
        return (signal - mean)/standard_dev

    def load_read(self, read_id):
        with h5py.File(self._filename, 'r') as h5file:
            read = h5file['Reads'][read_id]
            DAC = self._normalize_signal(list(read['Dacs'][()]))
            RTS = deque(read['Ref_to_signal'][()])
            REF = deque(read['Reference'][()])
        return DAC, RTS, REF
    
    def load_read_label_length(self, read_id):
        with h5py.File(self._filename, 'r') as h5file:
            return len(h5file['Reads'][read_id]['Reference'][()])

    def load_read_ids(self):
        with h5py.File(self._filename, 'r') as h5file:
           return list(h5file['Reads'].keys())
