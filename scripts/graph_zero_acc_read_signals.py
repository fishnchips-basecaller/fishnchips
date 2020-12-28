import argparse
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

path = '/nvme1/fishnchips/trained_models/chiron_data_run/fishnchips08_250_5CNN_25H_4B_6MPK_evaluation.json'

def get_file_data(filepath, filename):
    assert filename.split('.')[1] == 'json'
    path = f'{filepath}/{filename}'
    with open(path, 'r') as f:
        data = json.load(f)
        return pd.DataFrame(data)

def get_unmatched_read_ids(df):
    df = df[df["cigacc"]==0]
    return df['read_id']

def get_matched_read_ids(df):
    df = df[df["cigacc"]!=0]
    return df['read_id']

def get_signals(df):
    dacs = {}
    for e in df:
        path = f'data/eval/{e}.signal'
        with open(path, 'r') as f:
            signal_str = f.read()
            dac = list(map(int, signal_str.split(' ')))
            dacs[e] = normilize_signal(dac)
    return dacs

def normilize_signal(signal):
        signal = np.array(signal).astype(np.int32)
        return signal - np.mean(signal)/np.std(signal)

def vizualize_signals(signals):
    for k in signals.keys():
        signal = signals[k]
        for i in range(0,len(signal), 100):
            if len(signal) < i + 1000:
                plt.plot(signal[i:])
            else:
                plt.plot(signal[i:i+1000])
            plt.show(block=False)
            plt.pause(0.001)
            plt.close()

def vizualize_signal_ends(signals):
    for k in signals.keys():
        signal = signals[k]
        plt.plot(signal[:2000])
        plt.show(block=False)
        plt.pause(2)
        plt.close()
        plt.plot(signal[-2000:])
        plt.show(block=False)
        plt.pause(2)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('--filepath', type=str, default="temps")
    args = parser.parse_args()

    df = get_file_data(args.filepath, args.filename)
    df = get_matched_read_ids(df)
    signals = get_signals(df)
    vizualize_signal_ends(signals)

