import os
import numpy as np
import pickle as pk
import pandas as pd
import cupy as cp
import sys
import joblib
from lasso import *

sys.path.append('../')
from estimation import *

def run_for_importance(dimw, feature_size_list, results_path, device):
    dimz=10
    df = pd.read_csv(f'{data_prefix}/full_dataset_dimw{dimw}_dimz{dimz}.csv')
    
    full_W_cols = [f'W{i+1}' for i in range(dimw)] 
    Z_cols = [f'Z{i+1}' for i in range(dimz)] 

    for reduced_w_dim in feature_size_list:
        reduced_W_cols, lasso_model = run_lasso(df, full_W_cols, 'Y', reduced_w_dim)
        joblib.dump(lasso_model, f"lasso_outputs/lasso_model_dimw{dimw}_reduceddim{len(reduced_W_cols)}_dimz{dimz}.pkl")
        ns = [10000, 50000, 75000, 100000]
        dim_dict = {'X': 1, 'Y': 1, 'W': reduced_w_dim, 'Z':10, 'full_W': dimw}
        Z_cols = [f'Z{i+1}' for i in range(dim_dict['Z'])] 

        reduced_df = df[reduced_W_cols+Z_cols + ['X', 'Y']]
        print('Starting to run causal estimates for', reduced_w_dim, dimw)
        results_df, submodel_performances_df = run_causal_estimates(reduced_df, ns, reduced_W_cols, Z_cols, dim_dict, device, results_path)
        results_df.to_csv(f'{results_path}/estimated_effects_dimw{dimw}_dimreduced{reduced_w_dim}_dimz{len(Z_cols)}.csv')
        submodel_performances_df.to_csv(f'{results_path}/submodel_performances_dimw{dimw}_dimreduced{reduced_w_dim}_dimz{len(Z_cols)}.csv')

if __name__ == '__main__':
    data_prefix = 'PATH'
    cp.cuda.Device(1).use()
    device = 'cuda:1'
    results_path = 'PATH'

    run_for_importance(dimw = 750, feature_size_list = [150, 300, 375, 450, 600], results_path = results_path, device = device)
    run_for_importance(dimw = 1000, feature_size_list = [200, 400, 500, 600, 800], results_path = results_path, device = device)
    run_for_importance(dimw = 1250, feature_size_list = [250, 500, 625, 750, 1000], results_path = results_path, device = device) 
    run_for_importance(dimw = 1500, feature_size_list = [300, 600, 750, 900, 1200], results_path = results_path, device = device) 
    run_for_importance(dimw = 1750, feature_size_list = [350, 700, 875, 1050, 1400], results_path = results_path, device = device)

