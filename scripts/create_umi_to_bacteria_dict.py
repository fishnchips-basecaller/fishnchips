import json
from sortedcontainers import SortedDict

umireflinkcsvfile = "/path/to/foldertobacteriamapping.csv"
fast5folder = "/path/to/fast5"

umidict = {}

with open(umireflinkcsvfile, "r") as f:
    for i,line in enumerate(f):
        print(f"{i:05d}", end="\r")
        umifolder, bact = line.replace("\n", "").split(",")
        with open(f"{fast5folder}/{umifolder}/filename_mapping.txt", "r") as g:
            for sline in g:
                uid = sline.split("\t")[0]
                umidict[uid] = bact

sd = SortedDict(umidict)
with open("../temps/uids.json", "w") as f:
    json.dump(sd, f)

# %%
