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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

sys.path.append('utils')
import models
from losses import *
from train_utils import *
from training import *

seed_value = 35
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)

def greedy_top_feats(df, k, lambda_reg):
    df_copy = df.copy()
    # Score = Importance - (Lambda * Bias)
    df_copy['score'] = df_copy['norm_importance'] - (lambda_reg * df_copy['norm_bias'])
    
    # Pick Top K
    top_k = df_copy.nlargest(k, 'score')
    
    return top_k['norm_importance'].sum(), top_k['norm_bias'].sum()

def analyze_tradeoff(df, k, num_lambdas=500, max_lambda=10.0):
    # Sweep Lambda
    lambdas = np.linspace(0, max_lambda, num_lambdas)
    results = []
    
    for l in lambdas:
        imp, bias = greedy_top_feats(df, k, l)
        results.append({'lambda': l, 'total_importance': imp, 'total_bias': bias})
        
    results_df = pd.DataFrame(results)
    
    # Geometric Knee Finding
    # Define vector from Start (L=0) to End (L=Max)
    p1 = results_df.iloc[0][['total_bias', 'total_importance']].values
    p2 = results_df.iloc[-1][['total_bias', 'total_importance']].values
    
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    # Perpendicular vector
    vec_perp = np.array([-line_vec_norm[1], line_vec_norm[0]])
    
    # Calculate distance of every point to the line
    distances = []
    for i, row in results_df.iterrows():
        point = row[['total_bias', 'total_importance']].values
        dist = np.dot(point - p1, vec_perp)
        distances.append(dist)
        
    results_df['distance'] = distances
    
    # The knee is the point with max distance from the baseline
    knee_point = results_df.loc[results_df['distance'].abs().idxmax()]
    
    return knee_point, results_df

def get_data_columns(result_bias_importance, disease, dimreduced):
    knee, df_tradeoff = analyze_tradeoff(result_bias_importance, k=dimreduced)
    plt.figure(figsize=(10, 6))
    plt.plot(df_tradeoff['total_bias'], df_tradeoff['total_importance'], color='blue', alpha=0.6)

    # Scatter plot to show lambda progression
    sc = plt.scatter(df_tradeoff['total_bias'], df_tradeoff['total_importance'], 
                    c='blue', s=30)

    # Highlight Knee
    plt.scatter(knee['total_bias'], knee['total_importance'], 
                color='red', s=150, zorder=5, edgecolors='white', 
                label=f'Elbow (Lambda={knee["lambda"]:.2f})')

    # Baseline reference line
    p1 = df_tradeoff.iloc[0]
    p2 = df_tradeoff.iloc[-1]
    plt.plot([p1['total_bias'], p2['total_bias']], 
            [p1['total_importance'], p2['total_importance']], 
            'k--', alpha=0.5)

    plt.title(f'Feature Selection Trade-off: Bias vs Importance, k = {dimreduced}')
    plt.xlabel('Total Bias (Lower is Better)')
    plt.ylabel('Total Importance (Higher is Better)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(f'greedy_selection_figs/{disease}/greedy_featureselection_elbowplot_k{dimreduced}.png')

    lmbda = knee['lambda']
    print(f'Lambda: {lmbda}')
    result_bias_importance['score'] =  result_bias_importance['norm_importance'] - (lmbda * result_bias_importance['norm_bias'])
    result_bias_importance.sort_values('score', ascending = False, inplace = True)
    print(result_bias_importance[['norm_bias', 'score', 'norm_importance']])
    selected_feats = list(result_bias_importance.nlargest(dimreduced, 'score')['index'])
    return selected_feats


# set up dataset
def main(disease, dimreduced):
    print(f'Running {disease} {dimreduced}')
    path = f'DATASET PATH'

    device = torch.device("cpu")
    save_directory = f'MODEL PATH'
    os.makedirs(save_directory, exist_ok=True)

    # import data
    data_df = pd.read_csv(f'{path}/12_18_{disease}_llama_features.csv', index_col = 0)

    # decide which columns to keep
    result_bias_importance = pd.read_csv(f'{path}/12_23_{disease}_bias_importance_features.csv')
    scaler = MinMaxScaler()
    result_bias_importance[['norm_importance', 'norm_bias']] = scaler.fit_transform(result_bias_importance[['abs_weight','abs_smd']])
    data_columns = get_data_columns(result_bias_importance, disease, dimreduced)

    # now run the rest of the stuff
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
    # weight_y = 0.01 # set to something bigger for SLE
    weight_y = sum(data_df.loc[train_mask, 'boolean_value'])/len(data_df.loc[train_mask])
    print(weight_y)
    weight_prev = torch.tensor(np.asarray([1/(2*(1-weight_y)), 1/(2*weight_y)])).to(device)
    double_weight_prev = torch.tensor(np.asarray([1/(4*(1-weight_y)), 2/(2*weight_y)])).to(device)


    grid_search_all = {'learning_rate': [1e-3, 1e-4, 1e-5],
                        'weight_decay': [1e-2, 1e-3],
                'optimizer':['AdamW'],
                'loss_weights': [unweighted_weights, weight_prev, double_weight_prev],
                'prior_prob': [None] 
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

main('ami', 700)
main('ami', 600)
main('ami', 500)
main('ami', 400)
main('ami', 300)
