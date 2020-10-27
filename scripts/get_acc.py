import pandas as pd
import json

path = '/nvme1/fishnchips/trained_models/chiron_data_run/fishnchips08_250_5CNN_25H_4B_6MPK_evaluation.json'

with open(path, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(df)
print(f'ACC with unmatched:{df["cigacc"].mean()}')
df_m = df[df['cigacc'] != 0]
print(f'ACC without unmatched:{df_m["cigacc"].mean()}')
print(f'Number of unmatched:{len(df[df["cigacc"]==0])}/{len(df)}')
