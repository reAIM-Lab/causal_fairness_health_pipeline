import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import torch 
import sys
import pickle


def dict_to_dfs(results_path, data_columns):
    results_dict = torch.load(results_path, weights_only = False)
    # Unpack dictionary
    pids_list = results_dict["pids"]
    time_iter_list = results_dict["time_iter"]
    imp_list = results_dict["importance_scores"]
    rank_list = results_dict["ranked_feats"]

    rows_importance = []
    rows_ranked = []

    for pids, titer, imps, ranks in zip(pids_list, time_iter_list, imp_list, rank_list):
        # convert tensors to numpy
        pids = np.asarray(pids)
        titer = np.repeat(titer, len(pids))  # broadcast single time_iter

        # Build DataFrames directly
        df_imp = pd.DataFrame(imps, columns=data_columns)
        df_rank = pd.DataFrame(ranks, columns=data_columns)

        meta = pd.DataFrame({
            "pid": pids,
            "time_iter": titer
        })

        rows_importance.append(pd.concat([meta, df_imp], axis=1))
        rows_ranked.append(pd.concat([meta, df_rank], axis=1))

    # Concatenate all batches
    df_importance = pd.concat(rows_importance, ignore_index=True)
    print(df_importance.shape)
    df_ranked = pd.concat(rows_ranked, ignore_index=True)

    return df_importance, df_ranked

if __name__ == "__main__":
    """
    PATHS
    """    
    with open(f'{int_data_path}/COLUMNS', "rb") as fp:   # Unpickling
        data_columns = pickle.load(fp)
    data_columns = data_columns['W']

    list_ranked = []
    list_importance = []
    for time in [10, 20, 30, 40, 50, 59]:
        print(time)
        results_path = f'{afo_data_path}/11_10_baseline_transformer_w_AFO_outputs_time{time}.pt'
        df_importance, df_ranked = dict_to_dfs(results_path, data_columns)
        list_ranked.append(df_ranked)
        list_importance.append(df_importance)

    df_ranked = pd.concat(list_ranked)
    df_importance = pd.concat(list_importance)
    print(df_importance.shape)
    
    print('saving full ranking and importance')
    df_ranked.to_csv(f'{afo_data_path}/11_10_baseline_transformer_w_AFO_complete_ranked_dfs.csv', index=False)
    df_importance.to_csv(f'{afo_data_path}/11_10_baseline_transformer_w_AFO_complete_importance_dfs.csv', index=False)
