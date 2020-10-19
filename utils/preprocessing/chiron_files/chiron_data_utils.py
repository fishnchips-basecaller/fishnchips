from collections import deque
from utils.Other import labelBaseReverseMap

def process_label_str(label_str, as_bases=False):
    ref = []
    rts = []
    
    label = label_str.split('\n')[:-1]
    for base_obj in label:
        split = base_obj.split(' ')
        rts.append(int(split[0]))

        if as_bases:
            ref.append(split[2])
            continue
        ref.append(labelBaseReverseMap[split[2]])

    rts.append(int(label[-1].split(' ')[1]))
    return deque(ref), deque(rts)