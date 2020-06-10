#%%
import mappy as mp
import matplotlib.pyplot as plt
import numpy as np
import json
import re
import time, datetime

from gravlax.GravlaxBuilder import gravlax_for_file
from utils.assembler import assemble
from utils.preprocessing.DataGenerator import DataGenerator
from utils.preprocessing.DataPrepper import DataPrepper
from utils.Other import analyse_cigar, labelBaseMap, set_gpu_growth

set_gpu_growth()
aligner = mp.Aligner("temps/ref-uniq.fa")
if not aligner: raise Exception("ERROR: failed to load/build index")

class style():
    RED = lambda x: f"\033[31m{x}\033[0m"
    GREEN = lambda x: f"\033[32m{x}\033[0m"


num_trained = 7 # hardcoded based on training (number of bacteria it was trained on)
num_val = 1 # hardcoded based on training (number of bacteria it was validated on)
gravlax_model_name = "name of gravlax model as in path"
gravlax_train_time = "time of training as in path"
gravlax_weights_file = "h5 file inside the checkpoints folder"
model = f'trained_models/{gravlax_model_name}/{gravlax_train_time}/checkpoints/{gravlax_weights_file}.h5'

input_length = 300
reads_to_eval = 200


filename = "/ssd/mapped_umi11to5.hdf5"
bacteria = ["Escherichia", "Salmonella"]
generator = DataGenerator(filename, bacteria, batch_size=None, input_length=input_length, stride=20, reads_count=None, rnn_pad_size=None, use_maxpool=None).get_evaluate_batch()
modelname, gravlax = gravlax_for_file(input_length, model, num_trained, num_val, False, True)

result_dict = []
json_write_file = f"trained_models/{modelname}.json"

#%%

for idx in range(reads_to_eval):
    try:
        print(f"Evaluating {idx}/{reads_to_eval}...", end="")
        X, ref, raw, read_id = next(generator)

        start_time = time.time()

        prediction, logs = gravlax(X)
        assembled = assemble(prediction, window=7)
        try:
            # this crashes if no match found
            besthit = next(aligner.map(assembled))
            cigacc = 1-(besthit.NM/besthit.blen)
            result_dict.append({
                'read_id':read_id,
                'ctg': besthit.ctg,
                'r_st': besthit.r_st,
                'r_en': besthit.r_en,
                'NM': besthit.NM,
                'blen': besthit.blen,
                'cig': analyse_cigar(besthit.cigar_str),
                'cigacc': cigacc,
                'time': time.time()-start_time
            })
            print(style.GREEN(f"{modelname} ({cigacc*100:.2f})..."), end="")
        except:
            result_dict.append({
                'read_id':read_id,
                'ctg': 0,
                'r_st': 0,
                'r_en': 0,
                'NM': 0,
                'blen': 0,
                'cig': 0,
                'cigacc': 0,
                'time': time.time()-start_time
            })
            print(style.RED(f"{modelname}..."), end="")
        with open(json_write_file, 'w') as jsonfile:
            json.dump(result_dict, jsonfile)
        with open(f"trained_models/{modelname}_bench.fa", 'a') as f:
            f.write(f"@{read_id};{round(time.time())};{datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n")
            f.write(f"{assembled}\n")
        print("done.")
    except Exception as e:
        print(e)

# with open(jsonfile, 'r') as jf:
#     results = json.load(jf)

# %%
