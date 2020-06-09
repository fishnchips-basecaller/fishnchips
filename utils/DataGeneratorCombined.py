from utils.RawReadGenerator import RawReadGenerator
from utils.BacteriaDataLoader import BacteriaDataLoader
from collections import deque
import numpy as np

class DataGeneratorCombined(RawReadGenerator):
    def __init__(self,matched_reads_filename, root_folder, window_size, training, stride=30):
        super().__init__(root_folder, window_size, stride=30) 
        self._loader = BacteriaDataLoader(matched_reads_filename)
        self.skip_count = 0
        self.match_count = 0


    def generator(self):
        available_read_ids = self._loader.load_read_ids()    
        dac_gen = self._get_dac()
        for read_id, dac in dac_gen:
            if read_id not in available_read_ids:
                raw_windows = self._compute_raw_windows(dac)
                self.skip_count += 1
                yield (read_id, raw_windows, None, dac)                
                continue

            raw_windows = self._compute_raw_windows(dac)
            windows = self._compute_windows(read_id)
            self.match_count += 1
            yield (read_id, raw_windows, windows, dac)
        
    def _compute_raw_windows(self, dac):
        windows = []
        for i in range(0, len(dac)-self._window_size, self._stride):
            windows.append(dac[i:i+self._window_size])
        if i+self._window_size != len(dac):
            windows.append(dac[-self._window_size:])
        return np.array(windows).reshape((-1, self._window_size, 1))

    def _compute_windows(self, read_id):
        windows = []
        DAC, RTS, REF = self._loader.load_read(read_id)

        curdacs  = deque( [[x] for x in DAC[RTS[0]:RTS[0]+self._window_size-self._stride]], self._window_size )
        curdacts = RTS[0]+ self._window_size-self._stride
        labels  = deque([])
        labelts = deque([])
        
        while RTS[0] < curdacts:
            labels.append(REF.popleft())
            labelts.append(RTS.popleft())

        while curdacts+self._stride < RTS[-1]-self._window_size:
            curdacs.extend([[x] for x in DAC[curdacts:curdacts+self._stride]])
            curdacts += self._stride

            while RTS[0] < curdacts:
                labels.append(REF.popleft())
                labelts.append(RTS.popleft())

            while len(labelts) > 0 and labelts[0] < curdacts - self._window_size:
                labels.popleft()
                labelts.popleft()

            if len(labels) >= 1:
                windows.append(list(curdacs))
            
        return np.array(windows)