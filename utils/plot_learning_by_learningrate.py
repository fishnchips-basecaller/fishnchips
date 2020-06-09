#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

font = {'family' : 'serif',
        'size'   : 30}

rc('font', **font)
rc('text', usetex=True)

# lr1 = "/mnt/nvme/bio/mastersthesis/src/trained_models_2/15052020-1118/fishnchips71_250_5CNN_25H_4B_3MPK.npy"
# lr100 = "/mnt/nvme/bio/mastersthesis/src/trained_models_2/fish071-82-250-25H-5CNN-4B-3MPK/fishnchips71_250_5CNN_25H_4B_3MPK.npy"

lr1 = "/mnt/nvme/bio/mastersthesis/src/trained_models_2/fishP-lr1/fishnchips71_250_5CNN_25H_4B_6MPK.npy"
lr100 = "/mnt/nvme/bio/mastersthesis/src/trained_models_2/fishP-lr100/fishnchips71_250_5CNN_25H_4B_6MPK.npy"

nlr1 = np.load(lr1)
nlr100 = np.load(lr100)

# %%

# time starting at 0
nlr1[:,3] = nlr1[:, 3] - nlr1[0][3]
nlr100[:,3] = nlr100[:, 3] - nlr100[0][3]


# %%

plt.figure(figsize=(10,6))
plt.plot(nlr100[:, 2], label=r"FishNChips P $\eta=100$")
plt.plot(nlr1[:,2], label=r"FishNChips P $\eta=1$")
plt.xlabel("Epoch")
plt.ylabel("Average edit distance")
plt.legend()
plt.tight_layout()
plt.savefig("/tmp/fast_valacc.png")

# %%

