import numpy as np
import os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import sys
import gc
import pickle 
import torch
from joblib import dump, load
import json
import polars as pl

sys.path.append('../')
from preprocessing_utils import *

# set up dataset
dataset = 'mdcd_3yrs'
dataset_prefix = f'DATASET_PREFIX'
deeplearning_prefix = f'da_dl_'
path = 'PATH'
data_path = f'PATH'
int_path = f'PATH'
save_prefix = f'PATH'

make_scaler = True

# censor date to cohort start date
num_days_prediction = 90
column_dict = {}
column_dict['X'] = ['is_White', 'is_Black', 'is_MissingRace', 'is_Male', 'is_Female']

with open(f'{int_path}/{dataset_prefix}du_snomed_colnames', "rb") as fp:   #Pickling
    data_columns = pickle.load(fp)
print(len(data_columns))
column_dict['W'] = [i for i in data_columns if '_num_visits' not in i]
column_dict['Z'] = [i for i in data_columns if '_num_visits' in i]

with open(f'{int_path}/{save_prefix}colnames_dict', "wb") as fp:   #Pickling
    pickle.dump(column_dict, fp)
print(len(column_dict['W']), len(column_dict['Z']))


data_columns_xrace = {'X': ['is_White', 'is_Black', 'is_MissingRace'], 'Z': column_dict['Z'] + ['is_Male', 'is_Female'],
                       'W': column_dict['W']}
with open(f'{int_path}/{save_prefix}colnames_dict_xrace', "wb") as fp:   #Pickling
    pickle.dump(data_columns_xrace, fp)

"""
Load in a reduced set of columns here for dimensionality reduction
"""

# get split df
df_split = pd.read_csv(f'{int_path}/tvt_split_stratified_2dx.csv', index_col=0)
train_pids = list(df_split.loc[df_split['split']=='train', 'person_id'])
val_pids = list(df_split.loc[df_split['split']=='val', 'person_id'])
test_pids = list(df_split.loc[df_split['split']=='test', 'person_id'])
print(len(train_pids)/len(df_split), len(val_pids)/len(df_split), len(test_pids)/len(df_split))

# load in actual data
df_pl = pl.read_csv(f'{int_path}/{dataset_prefix}snomed_data.csv')
df_all_iters = df_pl.to_pandas()

df_iter_dates = pd.read_csv(f'{int_path}/{dataset_prefix}iteration_dates.csv')
df_pop = pd.read_csv(f'{data_path}/population_2dx.csv', parse_dates = ['psychosis_diagnosis_date', 'scz_diagnosis_date', 'cohort_start_date'])
df_pop.loc[df_pop['person_id'].isin(set(df_all_iters['person_id']))]


df_pop['is_White'] = 0
df_pop.loc[df_pop['race_concept_id']==8527, 'is_White'] = 1
df_pop['is_Black'] = 0
df_pop.loc[df_pop['race_concept_id']==8516, 'is_Black'] = 1
df_pop['is_MissingRace'] = 0
df_pop.loc[df_pop['race_concept_id']==0, 'is_MissingRace'] = 1
df_pop['is_Male'] = 0
df_pop.loc[df_pop['gender_concept_id']==8507, 'is_Male'] = 1
df_pop['is_Female'] = 0
df_pop.loc[df_pop['gender_concept_id']==8532, 'is_Female'] = 1

# checking the data to make sure all iterations are present
print(df_all_iters.isna().sum().sum())
min_iteration = df_all_iters['iteration'].min()
max_iteration = df_all_iters['iteration'].max()

ind_iterations = np.arange(min_iteration, max_iteration+1, 1)
print(min_iteration, max_iteration, len(ind_iterations))

df_all_iters['ranked_iteration'] = df_all_iters['iteration'] - min_iteration

df_all_iters.set_index(['person_id','ranked_iteration'], inplace=True)
df_all_iters.sort_index(inplace=True)
print('Check largest difference', find_largest_diff(df_all_iters)['largest_diff'].max()) # should be 1
df_all_iters.drop('iteration', axis=1, inplace=True)

# separate the actual data points into X, W, Z
df_pop.set_index('person_id', inplace=True)
train_data_X = df_pop.loc[train_pids, column_dict['X']]
val_data_X = df_pop.loc[val_pids, column_dict['X']]
test_data_X = df_pop.loc[test_pids, column_dict['X']]
print(train_data_X.shape, val_data_X.shape, test_data_X.shape)

train_data_Z = df_all_iters.loc[train_pids, column_dict['Z']]
val_data_Z = df_all_iters.loc[val_pids, column_dict['Z']]
test_data_Z = df_all_iters.loc[test_pids, column_dict['Z']]
print(train_data_Z.shape, val_data_Z.shape, test_data_Z.shape)

train_data_W = df_all_iters.loc[train_pids, column_dict['W']]
val_data_W = df_all_iters.loc[val_pids, column_dict['W']]
test_data_W = df_all_iters.loc[test_pids, column_dict['W']]
print(train_data_W.shape, val_data_W.shape, test_data_W.shape)

if make_scaler:
    # scale Z
    scaler_Z = StandardScaler()
    train_data_Z_mat = scaler_Z.fit_transform(train_data_Z)
    train_data_Z = pd.DataFrame(train_data_Z_mat, index = train_data_Z.index, columns = column_dict['Z'])
    print('done with fit/first transform')
    val_data_Z = pd.DataFrame(scaler_Z.transform(val_data_Z), index = val_data_Z.index, columns = column_dict['Z'])
    test_data_Z = pd.DataFrame(scaler_Z.transform(test_data_Z), index = test_data_Z.index, columns = column_dict['Z'])
    dump(scaler_Z, f'{int_path}/{save_prefix}{deeplearning_prefix}scaler_Z.bin', compress=True)

    # scale W
    scaler_W = StandardScaler()
    train_data_W_mat = scaler_W.fit_transform(train_data_W)
    train_data_W = pd.DataFrame(train_data_W_mat, index = train_data_W.index, columns = column_dict['W'])
    print('done with fit/first transform')
    val_data_W = pd.DataFrame(scaler_W.transform(val_data_W), index = val_data_W.index, columns = column_dict['W'])
    test_data_W = pd.DataFrame(scaler_W.transform(test_data_W), index = test_data_W.index, columns = column_dict['W'])
    dump(scaler_W, f'{int_path}/{save_prefix}{deeplearning_prefix}scaler_W.bin', compress=True)

else:
    scaler_Z = load(f'path to scaler')
    train_data_Z = pd.DataFrame(scaler_Z.transform(train_data_Z), index = train_data_Z.index, columns = column_dict['Z'])
    val_data_Z = pd.DataFrame(scaler_Z.transform(val_data_Z), index = val_data_Z.index, columns = column_dict['Z'])
    test_data_Z = pd.DataFrame(scaler_Z.transform(test_data_Z), index = test_data_Z.index, columns = column_dict['Z'])

    # scale W
    scaler_W = load(f'path to scaler')
    train_data_W = pd.DataFrame(scaler_W.transform(train_data_W), index = train_data_W.index, columns = column_dict['W'])
    val_data_W = pd.DataFrame(scaler_W.transform(val_data_W), index = val_data_W.index, columns = column_dict['W'])
    test_data_W = pd.DataFrame(scaler_W.transform(test_data_W), index = test_data_W.index, columns = column_dict['W'])

"""
### Pad the data
- Array of patient IDs (pids x 1) 
- Padded array (X): pids x time sequence x features (there should be X, W, Z)
- Mask: Binary pids x time sequence; 1 if that time is observed, 0 otherwise 
- Y: Event, binary (pids x 1)
"""
del df_all_iters
del df_pl
gc.collect()

datasets = Dataset_Object(df_pop.reset_index(), column_dict, min_iteration*-1, ind_iterations)
val_dataset = datasets.create_dataset_object(val_pids, val_data_X, val_data_Z, val_data_W)
test_dataset = datasets.create_dataset_object(test_pids, test_data_X, test_data_Z, test_data_W)
train_dataset = datasets.create_dataset_object(train_pids, train_data_X, train_data_Z, train_data_W)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle = True, pin_memory = True, num_workers = 4, persistent_workers = True)
torch.save(train_loader, f'{int_path}/{save_prefix}{deeplearning_prefix}train_loader.pth')

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle = True, pin_memory = True, num_workers = 4, persistent_workers = True)
torch.save(val_loader, f'{int_path}/{save_prefix}{deeplearning_prefix}val_loader.pth')

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle = True, pin_memory = True, num_workers = 4, persistent_workers = True)
torch.save(test_loader, f'{int_path}/{save_prefix}{deeplearning_prefix}test_loader.pth')
