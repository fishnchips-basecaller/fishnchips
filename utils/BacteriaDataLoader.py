import h5py
import numpy as np
from collections import deque

from utils.Other import get_taiyaki_filepath
from utils.cigar_vizualiser import get_mapping
from utils.Other import labelBaseMap

class BacteriaDataLoader:
    def __init__(self, filename):
        self._filename = get_taiyaki_filepath(filename)

    def load_read_ids(self):
        with h5py.File(self._filename, 'r') as h5file:
            return list(h5file['Reads'].keys())   

    def load_read(self, read_id):
        with h5py.File(self._filename, 'r') as h5file:
            read = h5file['Reads'][read_id]
            DAC = self._normalize_signal(list(read['Dacs'][()]))
            RTS = deque(read['Ref_to_signal'][()])
            REF = deque(read['Reference'][()])
        return DAC, RTS, REF

    def is_read_id_in_bacteria_lst(self, read_id, bacteria_lst):
        if len(bacteria_lst) == 0:
            return True
        bacteria_name = self._get_bacteria(read_id).lower()
        if bacteria_name is None:
            return False
        
        for bacteria in bacteria_lst:
            if bacteria.lower() in bacteria_name:
                return True
        return False

    def _get_bacteria(self, read_id):
        reference = self._load_reference(read_id)
        if reference is None:
            return None
        
        reference_str = "".join([labelBaseMap[base_token] for base_token in reference])
        return self._get_bacteria_from_reference(reference_str)
                
    def _load_reference(self, read_id):
        try:
            with h5py.File(self._filename, 'r') as h5file:
                return h5file['Reads'][read_id]['Reference'][()]
        except Exception as _:
            print(f"Error occured while loading a reference. Skipping {read_id}")
            return None

    def _get_bacteria_from_reference(self, reference):
        try:
            mapping = get_mapping(reference)
            return mapping.ctg
        except Exception as _:
            print(f"Error occured while getting bacteria type from reference.")
            return None

    def _normalize_signal(self, signal):
        signal = np.array(signal)
        mean = np.mean(signal)
        standard_dev = np.std(signal)
        return (signal - mean)/standard_dev