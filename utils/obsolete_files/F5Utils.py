import h5py

def getReads(path):
    f = h5py.File(path, 'r')
    reads = []
    for r in f.keys():
        reads.append(f[r])
    basecoded = list(map(lambda x: x['Analyses']['Basecall_1D_000']['BaseCalled_template']['Fastq'][()], reads))
    raw = list(map(lambda x: x['Raw']['Signal'][()], reads))
    return zip(basecoded, raw)