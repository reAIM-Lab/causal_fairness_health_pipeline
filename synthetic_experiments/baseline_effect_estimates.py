import os
import numpy as np
import pickle as pk
from datasets import *
from estimation import *
import pandas as pd
import cupy as cp

data_prefix = 'PATH'
results_path = 'PATH'
cp.cuda.Device(3).use()
device = 'cuda:3'

list_tuples = [(10, 750), (30, 1000), (30, 1250), (30, 1500), (30, 1750)]

dim_dict = {'X': 1, 'Y': 1}
ns = [10000, 50000, 75000, 100000]
for t in list_tuples:
    dimz, dimw = t
    dim_dict['Z'] = dimz
    dim_dict['W'] = dimw

    W_cols = [f'W{i+1}' for i in range(dimw)] 
    Z_cols = [f'Z{i+1}' for i in range(dimz)] 

    df = pd.read_csv(f'{data_prefix}/full_dataset_dimw{dimw}_dimz{dimz}.csv')
    results_df, submodel_performances_df = run_causal_estimates(df, ns, W_cols, Z_cols, dim_dict, device, 'results/estimates_xgb')
    
    results_df.to_csv(f'{results_path}/estimated_effects_dimw{dimw}_dimz{dimz}.csv')
    submodel_performances_df.to_csv(f'{results_path}/submodel_performances_dimw{dimw}_dimz{dimz}.csv')
