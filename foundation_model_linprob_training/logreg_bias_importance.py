import numpy as np
import torch
import torch.nn as nn
import scipy.stats as stats
import copy
import json
from pathlib import Path
from collections import defaultdict
import itertools
import pandas as pd
import sys
import pickle
from sklearn.preprocessing import StandardScaler

sys.path.append('../nontemporal_model_training/utils')
import models


# calculate feature bias
def check_bias(df_feats, data_cols, result_bias, subgroup1, subgroup2):
    data_cols.remove(subgroup1)
    data_cols.remove(subgroup2)
    df_feats_g1 = df_feats.loc[df_feats[subgroup1] == df_feats[subgroup1].max()] 
    df_feats_g2 = df_feats.loc[df_feats[subgroup2] == df_feats[subgroup2].max()] 
    print(len(df_feats_g1), len(df_feats_g2))

    n1, n2 = len(df_feats_g1), len(df_feats_g2)
    denominator = n1 + n2 - 2
    for feat in data_cols:
        # standardized mean difference
        mean1, mean2 = df_feats_g1[feat].mean(), df_feats_g2[feat].mean()
        var1, var2 = np.var(df_feats_g1[feat], ddof=1), np.var(df_feats_g2[feat], ddof=1)
        numerator = ((n1 - 1) * var1) + ((n2 - 1) * var2)
        pooled_std = np.sqrt(numerator / denominator)
        smd = (mean1 - mean2) / pooled_std
        result_bias.loc[feat, [f'smd', f'abs_smd']] = [smd, np.abs(smd)]
    return result_bias

def feature_importance(model, feature_columns):
    """
    Extracts weights from a PyTorch Logistic Regression model and saves them ranked by importance.
    """
    # 1. Get weights from the model
    # model.linear is your nn.Linear layer
    # .weight is a tensor of shape (1, n_features). We squeeze it to make it 1D.
    weights = model.linear.weight.detach().cpu().numpy().squeeze()
    print(len(weights))
    
    # 2. Create a DataFrame
    df_imp = pd.DataFrame({
        'feature': feature_columns,
        'weight': weights
    })
    
    # 3. Calculate 'importance' (Absolute value of weight)
    df_imp['abs_weight'] = df_imp['weight'].abs()
    df_imp.set_index('feature', inplace=True)
    return df_imp

def main(disease, model_path, group1, group2):
    print(disease)
    path = f'PATH'
    # import data
    data_df = pd.read_csv(f'{path}/12_18_{disease}_llama_features.csv', index_col = 0)
    all_columns = list(data_df.columns)
    embedding_cols = [i for i in all_columns if 'embedding_vec' in i]
    demo_cols = [i for i in all_columns if 'is_' in i]
    hcu_col = ['hcu']
    data_columns = embedding_cols + demo_cols + hcu_col

    # calculate importance
    model_config = torch.load(f"{model_path}/best_model_config.pt", weights_only=False)
    model = model_config['name'](**model_config['params']).to('cpu')
    model.load_state_dict(torch.load(f"{model_path}/best_model.pt", weights_only = False))
    results_importance = feature_importance(model, data_columns)

    # calculate bias
    scaler = StandardScaler()
    data_df = data_df.loc[data_df['split']=='train']
    data_df[data_columns] = data_df[data_columns].astype('float32')
    scaler.fit(data_df.loc[:, data_columns])
    data_df.loc[:, data_columns] = scaler.transform(data_df[data_columns])
    print('Done scaling features')

    result_bias = pd.DataFrame(columns = ['smd', 'abs_smd'])
    result_bias = check_bias(data_df, data_columns, result_bias, group1, group2)

    result_bias_importance = result_bias.merge(results_importance, how = 'inner', left_index=True, right_index=True).reset_index()
    print(len(result_bias_importance))
    result_bias_importance.to_csv(f'{path}/12_23_{disease}_bias_importance_features.csv')

main('sle', f'PATH', 'is_Male', 'is_Female')
main('ami', f'PATH', 'is_Male', 'is_Female')
main('t2dm', f'PATH', 'is_Black', 'is_White')