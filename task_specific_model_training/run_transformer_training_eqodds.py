import torch
import torch.nn as nn
import torch.optim as optim
import copy
import json
from pathlib import Path
from collections import defaultdict
import itertools
import pandas as pd
import sys
import pickle
from sklearn.metrics import *

sys.path.append('utils')
import models
from losses import *
from train_utils import *
from training import *

seed_value = 35
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)

"""
LOAD IN DATA AND SPECIFY MODEL PATH
"""

train_loader = torch.load(f'{int_path}/{dataset_prefix}train_loader.pth', weights_only=False)
val_loader = torch.load(f'{int_path}/{dataset_prefix}val_loader.pth', weights_only=False)
test_loader = torch.load(f'{int_path}/{dataset_prefix}test_loader.pth', weights_only=False)

# Define weights for loss function 
unweighted_weights = np.asarray([1, 1])

grid_search_all = {'learning_rate': [1e-3, 1e-4, 1e-5],
               'weight_decay': [1e-2, 1e-3],
               'optimizer':['AdamW', 'Adam', 'RMSprop'],
               'hidden_size': [128, 256, 512],
               'dim_feedforward': [128, 256, 512, 1024],
               'num_layers': [2, 4, 6],
               'dropout': [0.1, 0.3, 0.5],
               'emb_first': [True, False],
               'loss_weights': [unweighted_weights]
}

# baseline models: create config list and RANDOMLY SAMPLE
keys, values = zip(*grid_search_all.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
permutations_dicts = np.random.choice(permutations_dicts, size=30, replace=False).tolist()

for lmbda in [0.1, 1, 10, 100]: # 0.1, 1, 10, 100
    save_directory = f'SPECIFY PATH FOR MODEL'
    os.makedirs(save_directory, exist_ok=True)

    list_configs_baseline = []
    list_model_configs = []

    # print(f'Propensity AUROC: {roc_auc_score(px_z_df['race_Black'], px_z_df['px_z'])}')
    loss_regularization_dict = {'regularization_type': 'eqodds',
                                'x1': 'is_Black',
                                'x0': 'is_White',
                                'X_cols': data_columns_dict['X'],
                                'remove_col': ['is_MissingRace'],
                                'lambda': lmbda,
                                'device': device}

    for params in permutations_dicts:
        config = {
        "loss": {
            "type": "WeightedBCELoss",
            "weights": params['loss_weights']
        },
        "loss_regularization_dict": loss_regularization_dict,
        "optimizer": {
            "type": params['optimizer'],  
            "params": {"lr": params['learning_rate'], "weight_decay": params['weight_decay']}
        },
        "scheduler": {
            "type": "StepLR",
            "params": {"step_size": 5, "gamma": 0.5}
        },
        "early_stopping_patience": 5,
        "epochs": 30,
        "device":device,
        "model_selection": 'loss', # auroc or loss
        "init_model": False # False if model architecture is part of hyperparam search
    }
        list_configs_baseline.append(config)

        if params['emb_first'] == True:
            model_config = {'name': models.TransformerModelEmbPE}
        else: 
            model_config = {'name': models.TransformerModelPEEmb}    
        model_config['params'] = {'hidden_size': params['hidden_size'], 'dim_feedforward': params['dim_feedforward'],
                                'num_layers': params['num_layers'], 'num_heads': 4, 'dropout': params['dropout'],
                                'n_features': len(data_columns)}
        list_model_configs.append(model_config)

    torch.save(list_configs_baseline, f"{save_directory}/hparam_configs_list.pt")
    torch.save(list_model_configs, f"{save_directory}/model_configs_list.pt")

    search_hyperparams(list_configs_baseline, list_model_configs, train_loader, val_loader, test_loader, save_directory, device)

    # save final output
    model_config = torch.load(f"{save_directory}/best_model_config.pt", weights_only=False)
    model = model_config['name'](**model_config['params']).to(device)
    model.load_state_dict(torch.load(f"{save_directory}/best_model.pt", weights_only = False))

    config = torch.load(f"{save_directory}/best_hparam_config.pt", weights_only=False) 
    loss_config = config["loss"]
    criterion = get_loss_function(loss_config)

    _, train_output = test(train_loader, model, device, criterion, 
                            config.get("loss_regularization_dict"))
    _, val_output = test(val_loader, model, device, criterion, 
                            config.get("loss_regularization_dict"))
    _, test_output = test(test_loader, model, device, criterion, 
                            config.get("loss_regularization_dict"))
    pd.DataFrame(train_output, columns = ['person_id', 'y_true', 'y_pred']).to_csv(os.path.join(save_directory, "train_outputs.csv"))
    pd.DataFrame(val_output, columns = ['person_id', 'y_true', 'y_pred']).to_csv(os.path.join(save_directory, "val_outputs.csv"))
    pd.DataFrame(test_output, columns = ['person_id', 'y_true', 'y_pred']).to_csv(os.path.join(save_directory, "test_outputs.csv"))
