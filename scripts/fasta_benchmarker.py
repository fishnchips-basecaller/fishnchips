#%%

import numpy as np
import mappy as mp 
import os, sys
from collections import deque
import json
from utils.Other import analyse_cigar

aligner = mp.Aligner("temps/ref-uniq.fa")

class style():
    RED = lambda x: f"\033[31m{x}\033[0m"
    GREEN = lambda x: f"\033[32m{x}\033[0m"

if len(sys.argv)>1:
    experiments = [
        (sys.argv[1], sys.argv[2])
    ]
else:
    experiments = [
        # ('path', 'name'),
    ]

overwrite_all = False

read_dict_file = "temps/uids.json"
with open(read_dict_file) as f:
    read_dict = json.load(f)

for path,experiment_name in experiments:
    print(f"Running {experiment_name}")

    result_dict = []
    json_write_file = f"trained_models/fa_{experiment_name}.json"

    if os.path.isfile(json_write_file) and not overwrite_all:
        answer = input("File exists. Overwrite (yes/no/all) [Y/n/a]?:")
        if answer in "Nn" and answer is not "":
            continue
        if answer in "Aa" and answer is not "":
            overwrite_all = True

    reads = {}

    fastafiles = []

    if os.path.isfile(path):
        fastafiles.append(path)
    else:
        fastafiles.extend([os.path.join(path, x) for x in os.listdir(path) if x.split(".")[-1] in ["fastq", "fasta", "fa", "fq"]])

    

    for fastafile in fastafiles:
        with open(fastafile, "r") as f:
            data = deque(f.readlines())
        while len(data) > 0:
            l = data.popleft()
            if l[0] in '@>':
                rid = l[1:-1] # remove @ and \n
                rid = rid.split(" ")[0]
                rid = rid.split(";")[0]
                dna = ""
                while len(data) > 0 and len(data[0]) != 0 and data[0][0] not in ">@":
                    dna += data.popleft()[:-1]
                if "+" in dna:
                    dna = dna.split("+")[0]
                reads[rid] = dna

    cigaccs = []
    misses = 0
    tot_reads = len(reads)
    for i, (rid, dna) in enumerate(reads.items()):
        if len(dna) > 5000:
            continue
        if rid in read_dict.keys():
            print(f"Aligning {i:04d}/{tot_reads}"+" "*50, end="\r")
            found = False
            for hit in aligner.map(dna):
                if hit.ctg.split(";")[0] == read_dict[rid]:
                    found = True
                    cigacc = 1-(hit.NM/hit.blen)
                    result_dict.append({
                        'read_id':rid,
                        'dna_len': len(dna),
                        'ctg': hit.ctg,
                        'r_st': hit.r_st,
                        'r_en': hit.r_en,
                        'q_st': hit.q_st,
                        'q_en': hit.q_en,
                        'NM': hit.NM,
                        'blen': hit.blen,
                        'cig': analyse_cigar(hit.cigar_str),
                        'cigacc': cigacc
                    })
                    # print(style.GREEN(f"{modelname} ({cigacc*100:.2f})..."), end="")
                    cigaccs.append(cigacc*100)
                    break

            if not found:
                misses += 1
                result_dict.append({
                    'read_id':rid,
                    'dna_len': len(dna),
                    'ctg': 0,
                    'r_st': 0,
                    'r_en': 0,
                    'q_st': 0,
                    'q_en': 0,
                    'NM': 0,
                    'blen': 0,
                    'cig': 0,
                    'cigacc': 0
                })
                # print(style.RED(f"{rid}..."), end="")
                cigaccs.append(0)
            with open(json_write_file, 'w') as jsonfile:
                json.dump(result_dict, jsonfile, indent=4)
        else:
            print(f"{rid} not in dict")

    print(f"Average cigacc {np.mean(cigaccs):.2f}%, {misses} not found")
# %%
