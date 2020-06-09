#%%

import os
import sys
import re
import time
import datetime

from utils.RawReadGenerator import RawReadGenerator
from models.FishNChips import FishNChips

from utils.attention_evaluation_utils import build, evaluate_batch
from utils.assembler import assemble
from utils.Other import analyse_cigar, labelBaseMap, set_gpu_growth

set_gpu_growth()

in_folder = '../somedata/singlefast5/'
out_folder = 'fasta'

ATTENTION_BLOCKS = 4
CNN_BLOCKS = 5
MAXPOOL_BLOCK_IDX = 3
D_MODEL = 250
DFF = 512
NUM_HEADS = 25
ENCODER_MAX_LENGTH = 300
DECODER_MAX_LENGTH = 100
DROPOUT_RATE = 0.1
STRIDE = 30
MAX_POOL_KERNEL = 2

READS = 10
BATCH_SIZE = 64
AS_BASE_STRING = True

MODEL_IDENT = f"fishnchips_{D_MODEL}_{CNN_BLOCKS}CNN_{NUM_HEADS}H_{ATTENTION_BLOCKS}B"
if MAX_POOL_KERNEL != 2:
    MODEL_IDENT = f"{MODEL_IDENT}_{MAX_POOL_KERNEL}MPK"

MODEL_LOAD_FILENAME = f"./trained_models/{MODEL_IDENT}"

OUT_DIR = f"{out_folder}/{MODEL_IDENT}"


if os.path.isdir(OUT_DIR):
    answer = input("Basecall exists, overwrite [y/N]?:")
    if answer not in "yY" or answer == "":
        sys.exit(1)

    os.removedirs(OUT_DIR)

os.makedirs(OUT_DIR)

input_length = 300

readGeneratorObj = RawReadGenerator(in_folder, window_size=ENCODER_MAX_LENGTH, stride=STRIDE)
readGenerator = readGeneratorObj.generator()

def pretty_print_progress(current_begin, current_end, total):
    progstr = "["
    for i in range(0, total, total//100):
        if i>=current_begin and i<current_end:
            progstr += "x"
        else:
            progstr += "-"
    progstr += "]"
    return progstr

#%%

fish = FishNChips(
    num_cnn_blocks=CNN_BLOCKS,
    max_pool_layer_idx=MAXPOOL_BLOCK_IDX,
    max_pool_kernel_size=MAX_POOL_KERNEL,
    num_layers=ATTENTION_BLOCKS,
    d_model=D_MODEL,
    output_dim=1 + 4 + 1 + 1,  # PAD + ATCG + START + STOP
    num_heads=NUM_HEADS,
    dff=DFF,
    pe_encoder_max_length=ENCODER_MAX_LENGTH,
    pe_decoder_max_length=DECODER_MAX_LENGTH,
    rate=DROPOUT_RATE)

build(fish)
fish.load_weights(f"{MODEL_LOAD_FILENAME}.h5")

i = 0
for filename, dac in readGenerator:
    i+=1
    nr_windows = len(dac)

    y_pred = []
    for b in range(0,nr_windows,BATCH_SIZE):
        x_batch = dac[b:b+BATCH_SIZE]
        print(f"{i:03d}/{readGeneratorObj.len} Predicting windows {pretty_print_progress(b, b+len(x_batch), nr_windows)} {b:04d}-{b+len(x_batch):04d}/{nr_windows:04d}", end="\r")
        y_batch_pred, _ = evaluate_batch(x_batch, fish, len(x_batch), as_bases=AS_BASE_STRING)
        y_pred.extend(y_batch_pred)

    assembly = assemble(y_pred, window=7)

    with open(f"{OUT_DIR}/result.fa", 'a') as f:
        f.write(f"@{filename};{round(time.time())};{datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n")
        f.write(f"{assembly}\n")


# %%
