
import numpy as np
from os import listdir

def normilize_signal(signal):
    signal = np.array(signal).astype(np.int32)
    return (signal - np.mean(signal)/np.std(signal))

def load_file(filename):
    with open(filename, 'r') as f:
        return f.read()

def get_file_dict(dir):
    file_dict = {}

    for filename in listdir(dir):
        filepath = f"{dir}/{filename}"
        name, extension = filename.split(".")

        if name not in file_dict:
            file_dict[name] = {} 
        
        file_dict[name][extension] = filepath
    
    return file_dict

    data = []
    for i,k in enumerate(file_dict.keys()):
        print(f"fethich data: {i+1}/{len(file_dict.keys())}", end="\r")
        assert len(file_dict[k]) == 2, "Failed to construct data."
        label_str = load_file(file_dict[k]["label"])

        label = label_str.split('\n')[:-1]
        ref = []
        rts = []

        for base_obj in label:
            split = base_obj.split(' ')
            print(f"{split} | {base_obj}")
            rts.append(split[0])
            ref.append(split[2])
        rts.append(label[-1].split(' ')[1])
        print()
        print(len(ref))
        print(len(rts))

        return
        signal_str = load_file(file_dict[k]["signal"])
        signal = list(map(int, signal_str.split(' ')))
        signal_normalized = normilize_signal(signal)
        data.append((signal_normalized,label))

        if i == 10:
            return data
    return data   

dir = "./data/train"
filename = "./data/train/ecoli_0001.signal"
file_dict = get_file_dict(dir)



