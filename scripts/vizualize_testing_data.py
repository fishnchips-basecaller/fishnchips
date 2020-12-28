import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import deque
sys.path.insert(0,'../')


from utils.preprocessing.chiron_files.chiron_data_loader import ChironDataLoader

def fetch_read(read_id, loader, window_size=300, window_stride=30, min_labels_per_window=30):
        x_read = []
        y_read = []
        print(f"*** fetching id:{read_id}.")       
        DAC, RTS, REF = loader.get_read(read_id)
        
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

def vizualize_signal_ends(signal):
    for i in range(0,len(signal), 100):
        if len(signal) < i + 1000:
            plt.plot(signal[i:])
        else:
            plt.plot(signal[i:i+1000])
        plt.show(block=False)
        plt.pause(2)
        break
        # plt.close()

loader = ChironDataLoader('../data/eval')
ids = loader.get_ids()
x,y = fetch_read(ids[0], loader)

x = np.array(x)
y = np.array(y)

# print(x.shape)
print(y[0])