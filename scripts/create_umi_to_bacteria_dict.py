import json
from sortedcontainers import SortedDict

umireflinkcsvfile = "/mnt/nvme/bio/extracted/umi_ref_link.csv"
fast5folder = "/mnt/nvme/bio/extracted/fast5"
tot_folders = 23574 # number of lines in umi ref file

umidict = {}

with open(umireflinkcsvfile, "r") as f:
    for i,line in enumerate(f):
        print(f"{i:05d}/{tot_folders}", end="\r")
        umifolder, bact = line.replace("\n", "").split(",")
        with open(f"{fast5folder}/{umifolder}/filename_mapping.txt", "r") as g:
            for sline in g:
                uid = sline.split("\t")[0]
                umidict[uid] = bact

sd = SortedDict(umidict)
with open("../temps/uids.json", "w") as f:
    json.dump(sd, f)

# %%
