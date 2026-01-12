import os
import numpy as np
import pickle as pk
from datasets import *
from estimation import *
import pandas as pd
import cupy as cp

# create dataset
n = 100000
data_path = 'PATH'

# dimz, dimw = 10, 750
lambda_dict = {'X': 0.5, 'W': 300, 'Z': 0.5}
sigma_dict = {'U': 1, 'X': 0.5, 'W': 0.1, 'Y': 0.2}

dict_hidden_sizes = {'hidden_U_to_X': (16,), 'hidden_UZ_to_W': (256, 128),
                    'hidden_X_to_Y': (8,), 'hidden_W_to_Y': (128,), 
                    'hidden_Z_to_Y': (16,)}

starting_dimW = 750
dict_dims = {'Y': 1, 'X': 1, 'Z': 10, 'W': starting_dimW}
df, params = create_data(dict_dims, lambda_dict, sigma_dict, n=n, seed=31,
                            dict_hidden_sizes = dict_hidden_sizes)

gt_effects = compute_data_effects(df, 'X', 'Y')
EYx0, EYx1, EYx1Wx0, TV, TE, NDE, NIE, EYx0_obs, EYx1_obs, expse_x0, expse_x1, SE = gt_effects
with open(f'{data_path}/nn_parameters_dimw{dict_dims["W"]}_dimz{dict_dims["Z"]}.pkl', 'wb') as f:
    pk.dump(params, f)

df.to_csv(f'{data_path}/full_dataset_dimw{dict_dims["W"]}_dimz{dict_dims["Z"]}.csv')

dict_effects = {'EYx0': EYx0, 'EYx1': EYx1, 'EYx1Wx0': EYx1Wx0, 
                'TV': TV, 'TE': TE, 
                'NDE': NDE, 'NIE': NIE, 
                'EYx0_obs': EYx0_obs, 'EYx1_obs': EYx1_obs, 
                'expse_x0': expse_x0, 'expse_x1': expse_x1, 'SE': SE}
with open(f'{data_path}/true_effects_dimw{dict_dims["W"]}_dimz{dict_dims["Z"]}.pkl', 'wb') as f:
    pk.dump(dict_effects, f)


# Get Z = 30 datasets, based on W = 1000 MLPs
lambda_dict = {'X': 0.5, 'W': 300, 'Z': 0.5}
sigma_dict = {'U': 1, 'X': 0.5, 'W': 0.1, 'Y': 0.2}

dict_hidden_sizes = {'hidden_U_to_X': (16,), 'hidden_UZ_to_W': (256, 128),
                    'hidden_X_to_Y': (8,), 'hidden_W_to_Y': (128,), 
                    'hidden_Z_to_Y': (16,)}

starting_dimW = 1000
dict_dims = {'Y': 1, 'X': 1, 'Z': 30, 'W': starting_dimW}
df, params = create_data(dict_dims, lambda_dict, sigma_dict, n=n, seed=31,
                            dict_hidden_sizes = dict_hidden_sizes)

gt_effects = compute_data_effects(df, 'X', 'Y')
EYx0, EYx1, EYx1Wx0, TV, TE, NDE, NIE, EYx0_obs, EYx1_obs, expse_x0, expse_x1, SE = gt_effects
with open(f'{data_path}/nn_parameters_dimw{dict_dims["W"]}_dimz{dict_dims["Z"]}.pkl', 'wb') as f:
    pk.dump(params, f)

df.to_csv(f'{data_path}/full_dataset_dimw{dict_dims["W"]}_dimz{dict_dims["Z"]}.csv')

dict_effects = {'EYx0': EYx0, 'EYx1': EYx1, 'EYx1Wx0': EYx1Wx0, 
                'TV': TV, 'TE': TE, 
                'NDE': NDE, 'NIE': NIE, 
                'EYx0_obs': EYx0_obs, 'EYx1_obs': EYx1_obs, 
                'expse_x0': expse_x0, 'expse_x1': expse_x1, 'SE': SE}
with open(f'{data_path}/true_effects_dimw{dict_dims["W"]}_dimz{dict_dims["Z"]}.pkl', 'wb') as f:
    pk.dump(dict_effects, f)


for dimw in [1250, 1500, 1750]:
    dict_dims['W'] = dimw

    # read in params -- note that this reading in dimZ twice bc I want dimw=dimz = 30
    with open(f'{data_path}/nn_parameters_dimw{starting_dimW}_dimz{dict_dims["Z"]}.pkl', 'rb') as f:
        dict_prebuilt_mlps = pk.load(f)
    dict_prebuilt_mlps.pop('net_XZ_to_W', None)
    dict_prebuilt_mlps.pop('net_W_to_Y', None)

    dict_hidden_sizes['hidden_W_to_Y'] = (256,)
    df, params = create_data(dict_dims, lambda_dict, sigma_dict, n=n, seed=31,
                            dict_hidden_sizes = dict_hidden_sizes, dict_prebuilt_mlps = dict_prebuilt_mlps)

    gt_effects = compute_data_effects(df, 'X', 'Y')
    EYx0, EYx1, EYx1Wx0, TV, TE, NDE, NIE, EYx0_obs, EYx1_obs, expse_x0, expse_x1, SE = gt_effects

    with open(f'{data_path}/nn_parameters_dimw{dict_dims["W"]}_dimz{dict_dims["Z"]}.pkl', 'wb') as f:
        pk.dump(params, f)

    df.to_csv(f'{data_path}/full_dataset_dimw{dict_dims["W"]}_dimz{dict_dims["Z"]}.csv')

    dict_effects = {'EYx0': EYx0, 'EYx1': EYx1, 'EYx1Wx0': EYx1Wx0, 
                    'TV': TV, 'TE': TE, 
                    'NDE': NDE, 'NIE': NIE, 
                    'EYx0_obs': EYx0_obs, 'EYx1_obs': EYx1_obs, 
                    'expse_x0': expse_x0, 'expse_x1': expse_x1, 'SE': SE}
    with open(f'{data_path}/true_effects_dimw{dict_dims["W"]}_dimz{dict_dims["Z"]}.pkl', 'wb') as f:
        pk.dump(dict_effects, f)

