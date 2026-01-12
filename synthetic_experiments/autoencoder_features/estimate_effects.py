import os
import numpy as np
import pickle as pk
import pandas as pd
import cupy as cp
import sys
import scipy.stats as stats

sys.path.append('../')
from estimation import *

def run_for_autoencoder(dimw, feature_size_list, results_path, device):
    
    for reduced_dim in feature_size_list:
        dimz=10
        df = pd.read_csv(f'{data_prefix}/data_dimw{dimw}_reduced_dim{reduced_dim}_dimz{dimz}.csv')

        ns = [10000, 50000, 75000, 100000]
        dim_dict = {'X': 1, 'Y': 1, 'W': reduced_dim, 'Z':dimz, 'full_W': dimw}
        AE_cols = [f'AE_{i}' for i in range(dim_dict['W'])] 
        Z_cols = [f'Z{i+1}' for i in range(dim_dict['Z'])] 

        reduced_df = df[AE_cols+Z_cols + ['X', 'Y']]
        print('Starting to run causal estimates for', reduced_dim, dimw)
        results_df, submodel_performances_df = run_causal_estimates(reduced_df, ns, AE_cols, Z_cols, dim_dict, device, results_path)
        results_df.to_csv(f'{results_path}/estimated_effects_dimw{dimw}_dimreduced{reduced_dim}_dimz{len(Z_cols)}.csv')
        submodel_performances_df.to_csv(f'{results_path}/submodel_performances_dimw{dimw}_dimreduced{reduced_dim}_dimz{len(Z_cols)}.csv')


if __name__ == '__main__':
    data_prefix = 'PATH'
    results_path = 'PATH'
    cp.cuda.Device(3).use()
    device = 'cuda:3'
    run_for_autoencoder(dimw = 750, feature_size_list = [150, 300, 375, 450, 600], results_path = results_path, device = device)
    run_for_autoencoder(dimw = 1000, feature_size_list = [200, 400, 500, 600, 800], results_path = results_path, device = device) 
    run_for_autoencoder(dimw = 1250, feature_size_list = [250, 500, 625, 750, 1000], results_path = results_path, device = device) 
    run_for_autoencoder(dimw = 1500, feature_size_list = [300, 600, 750, 900, 1200], results_path = results_path, device = device)
    run_for_autoencoder(dimw = 1750, feature_size_list = [350, 700, 875, 1050, 1400], results_path = results_path, device = device)
