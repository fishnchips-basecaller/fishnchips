import tensorflow as tf
import os
import re
from datetime import datetime

labelBaseMap = {
    0: "A",
    1: "C",
    2: "G",
    3: "T",
    4: "-"
}

attentionLabelBaseMap = {
    0: 'P',
    1: 'A',
    2: 'C',
    3: 'G',
    4: 'T',
    5: 'S',
    6: 'E'
}

def print_tensor_to_file(value, file="info"):
    tf.print(value, output_stream=f"file://./{file}.out", summarize=1000000)

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


def get_taiyaki_filepath(filename):
    paths = ["/mnt/nvme/bio/taiyaki_aligned/",
             "/ssd/",
             "/user/student.aau.dk/fgravi18/data/",
             "/Users/felix/MsC/DNA/",
             "c:/Users/mirop/OneDrive/Documents/Programming/Data/bdm/",
             "/mnt/c/Users/mirop/OneDrive/Documents/Programming/Data/bdm/"]
    
    for path in paths:
        filepath = f"{path}{filename}"
        if os.path.isfile(filepath):
            print(f"Loading data from file:{filepath}")
            return filepath
    else:
        raise Exception(f"File {path}{filename} could not be found!")

def get_valid_taiyaki_filename(training):
    train_filenames = ["/mnt/nvme/bio/taiyaki_aligned/mapped_therest.hdf5",
			              "/user/student.aau.dk/fgravi18/data/mapped_therest.hdf5",
                          "/ssd/mapped_therest.hdf5",
                          "/Users/felix/MsC/DNA/mapped_therest.hdf5",
                          "c:/Users/mirop/OneDrive/Documents/Programming/Data/bdm/mapped_therest.hdf5",
                          "/mnt/c/Users/mirop/OneDrive/Documents/Programming/Data/bdm/mapped_therest.hdf5"]

    test_filenames = ["/mnt/nvme/bio/taiyaki_aligned/mapped_umi16to9.hdf5",
			              "/user/student.aau.dk/fgravi18/data/mapped_umi16to9.hdf5",
                          "/ssd/mapped_umi16to9.hdf5",
                          "/Users/felix/MsC/DNA/mapped_umi16to9.hdf5",
                          "c:/Users/mirop/OneDrive/Documents/Programming/Data/bdm/mapped_umi16to9.hdf5",
                          "/mnt/c/Users/mirop/OneDrive/Documents/Programming/Data/bdm/mapped_umi16to9.hdf5"]

    possible_filenames = train_filenames if training else test_filenames

    for filename in possible_filenames:
        if os.path.isfile(filename):
            print(f"Loading data from file:{filename}")
            return filename
    else:
        raise "Read data file could not be found!"

def get_valid_model_filename(model_name):
    possible_filenames = [
        f"/mnt/c/Users/mirop/OneDrive/Documents/Programming/Data/bdm/outputs/{model_name}/checkpoints/model.h5",
        f"c:/Users/mirop/OneDrive/Documents/Programming/Data/bdm/outputs/{model_name}/checkpoints/model.h5"
    ]

    print(possible_filenames[0])

    for filename in possible_filenames:
        if os.path.isfile(filename):
            return filename
    else:
        raise f"Model file could not be found!"
  
def analyse_cigar(cigar_string):
    res = re.findall(r'[\d]+[SMDI]', cigar_string)
    d = {"S":0,"M":0,"D":0,"I":0}
    for r in res:
        d[r[-1]] += int(r[:-1])
    return d

def with_timer(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        res = func(*args, **kwargs)
        end_time = datetime.now()
        took = end_time-start_time
        print(f"Took: {end_time-start_time}, Len: {len(res[0])}, Start:{start_time}, End:{end_time}, Class/function:{func.__qualname__ }")
        return res
    return wrapper

def with_eval_timer(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        res = func(*args, **kwargs)
        end_time = datetime.now()

        max_len = 0
        for seq in res[0]:
            if(len(seq) > max_len):
                max_len = len(seq)

        took = end_time-start_time
        avg_per_base = took / max_len
        print(f"Took: {end_time-start_time}, Max seq length: {max_len}, Avg time per base:{avg_per_base}, Class/function:{func.__qualname__ }")
        return res
    return wrapper
