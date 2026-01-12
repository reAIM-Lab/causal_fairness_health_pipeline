import os
import numpy as np
import pickle as pk
from estimation import *
import pandas as pd
import cupy as cp
import pickle 
import time
from sklearn.metrics import *

disease = 'T2DM' # AMI, SLE, SCZ
"""
LOAD IN DATA
"""


# set up columns! 
all_cols = list(df_data.columns)
cols_dict = {}
# cols_dict['Z'] = ['hcu', 'is_Black', 'is_White', 'is_MissingRace', 'is_OtherRace', 'is_Asian']
cols_dict['Z'] = ['hcu', 'is_Male']

X_col = 'is_Black'
Z_cols = cols_dict['Z']

print(len(df_data))
df_data = df_data.loc[(df_data['is_Black']==1) | (df_data['is_White']==1)]
print(len(df_data))

# Average over all px_z
print("Running inference for train/val/test splits...")

bootstraps = 100
K = 5
clip = 1e-2
cp.cuda.Device(0).use()
device = 'cuda:0'

list_results = []
for split_name in ["train", "tuning", "held_out"]:
    print(f"\nProcessing split: {split_name}")

    # Select the data for this split
    df_split = df_data[df_data["split"] == split_name].copy()
    n = len(df_split)
    all_idxs = np.arange(n)

    # Preallocate columns
    

    for i in tqdm(range(bootstraps), desc=f"Bootstraps ({split_name})"):
        folds = KFold(n_splits=K, shuffle=True, random_state=i)

        for k_split, (_, ts) in enumerate(folds.split(all_idxs)):
            file_path = os.path.join(model_path, f"models_boot{i}_split{k_split}.joblib")
            if not os.path.exists(file_path):
                print(f"Skipping missing model file: {file_path}")
                continue

            # Load saved model dictionary
            model_objects = joblib.load(file_path)

            # Subset this test fold
            df_ts = df_split.iloc[ts].copy()

            # Run predictions
            df_ts[f'px_z'] = pred(df_ts[Z_cols], model_objects["px_z"], clip=clip)

            mini_results = df_ts[['PID_unique', f'px_z', 'is_Black']]
            # Add back split name and person_id for tracking
            list_results.append(mini_results)

# Combine all splits
final_results = pd.concat(list_results, ignore_index=True)
print(len(final_results))
final_results = final_results.groupby('PID_unique').mean().reset_index()
print(len(final_results))
final_results.to_csv('SAVE PATH')

print(roc_auc_score(final_results['is_Black'], final_results['px_z']))
print(final_results.head())
