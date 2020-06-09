import tensorflow as tf
import os
import re

labelBaseMap = {
    0: "A",
    1: "C",
    2: "G",
    3: "T",
    4: "-"
}

def set_gpu_limit(limitMB):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limitMB)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        print(e)

def set_gpu_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
        
def get_valid_taiyaki_filename():
    possible_filenames = ["/mnt/nvme/taiyaki_aligned/mapped_umi16to9.hdf5",
                          "/hdd/mapped_therest.hdf5",
                          "/Users/felix/MsC/DNA/mapped_umi16to9.hdf5",
                          "c:/Users/mirop/OneDrive/Documents/Programming/Data/bdm/mapped_umi16to9.hdf5"]

    for filename in possible_filenames:
        if os.path.isfile(filename):
            return filename
    else:
        raise "No filename valid!"
        
def analyse_cigar(cigar_string):
    res = re.findall(r'[\d]+[SMDI]', cigar_string)
    d = {"S":0,"M":0,"D":0,"I":0}
    for r in res:
        d[r[-1]] += int(r[:-1])
    return d
