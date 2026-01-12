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
from sklearn.preprocessing import StandardScaler

sys.path.append('utils')
import models
from losses import *
from train_utils import *
from training import *

seed_value = 35
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)

# set up dataset
def main(disease, dimreduced):
    print(f'Running {disease} {dimreduced}')
    path = f'DATA PATH'

    device = torch.device("cpu")
    save_directory = f'MODEL PATH'
    os.makedirs(save_directory, exist_ok=True)

    # import data
    data_df = pd.read_csv(f'{path}/12_18_{disease}_llama_features.csv', index_col = 0)

    # decide which columns to keep
    result_bias_importance = pd.read_csv(f'{path}/12_23_{disease}_bias_importance_features.csv')
    result_bias_importance = result_bias_importance.sort_values('abs_smd', ascending=True)
    print(result_bias_importance.head())
    data_columns = list(result_bias_importance.iloc[0:dimreduced]['index'])
    print(data_columns[0:10])
    data_df[data_columns] = data_df[data_columns].astype('float32')

    scaler = StandardScaler()
    train_mask = data_df['split'] == 'train'
    scaler.fit(data_df.loc[train_mask, data_columns])
    data_df.loc[:, data_columns] = scaler.transform(data_df[data_columns])
    print('Done scaling features')

    # create tabular datasets
    train_dataset = TabularDataset(data_df, split='train', feature_cols=data_columns)
    val_dataset = TabularDataset(data_df, split='tuning', feature_cols=data_columns)
    test_dataset = TabularDataset(data_df, split='held_out', feature_cols=data_columns)

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16384, shuffle = True, pin_memory = True, num_workers = 4, persistent_workers = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16384, shuffle = True, pin_memory = True, num_workers = 4, persistent_workers = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16384, shuffle = True, pin_memory = True, num_workers = 4, persistent_workers = True)
    print(f"Features: {len(data_columns)}")

    # Define weights for loss function 
    unweighted_weights = torch.tensor(np.asarray([1, 1])).to(device)
    weight_y = 0.01 # set to something bigger for SLE
    #weight_y = sum(data_df.loc[train_mask, 'boolean_value'])/len(data_df.loc[train_mask])
    print(weight_y)
    weight_prev = torch.tensor(np.asarray([1/(2*(1-weight_y)), 1/(2*weight_y)])).to(device)
    double_weight_prev = torch.tensor(np.asarray([1/(4*(1-weight_y)), 2/(2*weight_y)])).to(device)


    grid_search_all = {'learning_rate': [1e-3, 1e-4, 1e-5],
                        'weight_decay': [1e-2, 1e-3],
                'optimizer':['AdamW'],
                'loss_weights': [unweighted_weights, weight_prev, double_weight_prev],
                'prior_prob': [weight_y] 
    }

    # baseline models: create config list 
    keys, values = zip(*grid_search_all.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # permutations_dicts = np.random.choice(permutations_dicts, size=30, replace=False).tolist()

    list_configs_baseline = []
    list_model_configs = []
    for params in permutations_dicts:
        config = {
        "loss": {
            "type": "WeightedBCELoss",
            "weights": params['loss_weights']
        },
        "optimizer": {
            "type": params['optimizer'],  
            "params": {"lr": params['learning_rate'], 'weight_decay': params['weight_decay']}
        },
        "scheduler": {
            "type": "StepLR",
            "params": {"step_size": 5, "gamma": 0.5}
        },
        "early_stopping_patience": 5,
        "epochs": 50,
        "device":device,
        "loss_regularization_dict": None,
        "model_selection": 'loss', # auroc or loss
        "init_model": False # False if model architecture is part of hyperparam search
    }
        list_configs_baseline.append(config)

        model_config = {'name': models.LogisticRegression}    
        model_config['params'] = {'input_dim': len(data_columns), 'prior_prob': params['prior_prob']}
        list_model_configs.append(model_config)

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
    pd.DataFrame(train_output, columns = ['PID_unique', 'person_id', 'pred_time', 'y_true', 'y_pred']).to_csv(os.path.join(save_directory, "train_outputs.csv"))
    pd.DataFrame(val_output, columns = ['PID_unique', 'person_id', 'pred_time', 'y_true', 'y_pred']).to_csv(os.path.join(save_directory, "val_outputs.csv"))
    pd.DataFrame(test_output, columns = ['PID_unique', 'person_id', 'pred_time', 'y_true', 'y_pred']).to_csv(os.path.join(save_directory, "test_outputs.csv"))

    test_output = pd.DataFrame(test_output, columns = ['PID_unique', 'person_id', 'pred_time', 'y_true', 'y_pred'])
    print(roc_auc_score(test_output['y_true'], test_output['y_pred']))

main('sle', 700)
main('sle', 600)
main('sle', 500)
main('sle', 400)
main('sle', 300)
# note: SLE should use weight_y = 0.01 = prior_prob (weight initialization) and batch size = 16k