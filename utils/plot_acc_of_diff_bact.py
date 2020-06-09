#%%

import json
import matplotlib.pyplot as plt
import numpy as np

# analysisfile = "/mnt/nvme/bio/mastersthesis/src/trained_models/fa_fishP-lr1-ass-allbact.json"
# analysisfile = "/mnt/nvme/bio/mastersthesis/src/trained_models/fa_fishP-lr1-concat-twobact.json"
# analysisfile = "/mnt/nvme/bio/mastersthesis/src/trained_models/fa_fishP-lr1-concat-allbact.json"

with open(analysisfile) as f:
    analysisdata = json.load(f)

analysisdata = [a for a in analysisdata if a['ctg'] != 0]

# %%

bacts = {}
longest_bact_name = 0
for c in analysisdata:
    bact = c['ctg'].split(";")[0]
    bact = c['ctg'].split("_")[0] # if only big bact
    if bact not in bacts:
        bacts[bact] = []
        longest_bact_name = len(bact) if len(bact) > longest_bact_name else longest_bact_name
    bacts[bact].append(c['cigacc'])


# %%

for bact, cigaccs in bacts.items():
    print(f"{bact: <{longest_bact_name}} : {np.mean(cigaccs)*100:.02f}% ({len(cigaccs)})")

# %%

outlier = "Escherichia"
trained = [c['cigacc'] for c in analysisdata if outlier not in c['ctg']]
untrained = [c['cigacc'] for c in analysisdata if outlier in c['ctg']]

print(f"trained acc  : {np.mean(trained)*100:.02f}%")
print(f"untrained acc: {np.mean(untrained)*100:.02f}%")

# %%

sbacts = sorted(bacts.keys())

plt.figure(figsize=(len(sbacts),7))
plt.bar(sbacts, [np.mean(bacts[b])*100 for b in sbacts])
plt.xticks(sbacts, rotation="vertical")
plt.xlabel("bacteria")
plt.ylabel("accuracy")
plt.tight_layout()
plt.savefig("per_bact_acc.png")

# %%
