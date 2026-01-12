import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

def pre_censor_data(df, df_pop, date_col):
    df = df.loc[df['person_id'].isin(df_pop['person_id'])]
    df = df.merge(df_pop[['person_id', 'censor_date']], how = 'inner', on = 'person_id')
    df = df.loc[df[date_col] <= df['censor_date']]
    return df

def drop_rare_occurrences(df, col_concept, col_id, size_pop, threshold):
    unique_occurrences = df[[col_id, col_concept]].drop_duplicates()
    unique_occurrences = unique_occurrences.value_counts(col_concept)
    common_occurrences = unique_occurrences[unique_occurrences/size_pop > threshold].index
    return df.loc[df[col_concept].isin(common_occurrences)]

def drop_unshared_features(df, col_concept, list_cols):
    df = df.loc[df[col_concept].isin(list_cols)]
    return df

def remove_highcorr_cols(threshold, list_colnames, df_corr):
    np.random.seed(43)
    upper_tri_corr = pd.DataFrame(index = df_corr.index, columns = df_corr.columns, data = np.triu(df_corr.values))
    melted_corr = df_corr.melt(ignore_index=False).reset_index()
    melted_corr = melted_corr.loc[melted_corr['index']!= melted_corr['variable']]
    melted_corr = melted_corr.loc[melted_corr['value']>=threshold]
    melted_corr.sort_values('value', ascending=False, inplace=True)

    feats_to_remove = []
    while(len(melted_corr)) > 0:
        ind = melted_corr.iloc[0]['index']
        var = melted_corr.iloc[0]['variable']
        num_index = len(melted_corr.loc[melted_corr['index']==ind]) + len(melted_corr.loc[melted_corr['variable']==ind])
        num_variable = len(melted_corr.loc[melted_corr['index']==var]) + len(melted_corr.loc[melted_corr['variable']==var])

        if num_index == num_variable:
            # remove a random variable
            rand_choice = (np.random.randint(0,2))
            if rand_choice == 0: 
                remove_var = ind
            else: 
                remove_var = var
        elif num_index < num_variable:
            # remove the "column"
            remove_var = var
        elif num_index > num_variable:
            # remove the "index"
            remove_var = ind

        # remove remove_var
        feats_to_remove.append(remove_var)
        melted_corr = melted_corr.loc[~(melted_corr['index'] == remove_var)]
        melted_corr = melted_corr.loc[~(melted_corr['variable'] == remove_var)]

    colnames_copy = list(list_colnames.copy())
    for i in feats_to_remove:
        colnames_copy.remove(i)
    return colnames_copy

def find_largest_diff(df):
    # Sort by pid and iteration
    df_sorted = df.sort_values(by=['person_id', 'iteration'])
    
    # Calculate the largest difference for each pid
    result = df_sorted.groupby('person_id')['iteration'].apply(
        lambda x: x.diff().max()
    ).reset_index(name='largest_diff')
    
    return result

class Dataset_Object:
    def __init__(self, df_pop, data_columns_dict, psychosis_iter, ind_iterations):
        self.df_pop = df_pop
        self.data_columns_dict = data_columns_dict
        self.psychosis_iter = psychosis_iter
        self.ind_iterations = ind_iterations

    def create_dataset_object(self, pids, data_X, data_Z, data_W):
        num_timesteps = len(self.ind_iterations)
    
        # get pids array
        pids_tensor = torch.from_numpy(np.asarray(pids))
        if np.asarray([pids_tensor[i] == pids[i] for i in range(len(pids))]).sum() != len(pids):
            print('PID Tensor does not match original pids')
            
        data_X.sort_index(inplace=True)
        data_W.sort_index(inplace=True)
        data_Z.sort_index(inplace=True)
    
        # get iterations for X tensor
        iterations = data_Z.reset_index()[['person_id', 'ranked_iteration']]
        iterations = iterations.groupby('person_id').agg(['min', 'max'])
        iterations = iterations.loc[pids]
        iterations = torch.from_numpy(np.asarray(iterations.values))
        
        # create tensors
        X_tensor = torch.zeros((len(pids_tensor), len(self.data_columns_dict['X'])))
        Z_tensor = torch.zeros((len(pids_tensor), num_timesteps, len(self.data_columns_dict['Z'])))
        W_tensor = torch.zeros((len(pids_tensor), num_timesteps, len(self.data_columns_dict['W'])))
        X_mask = torch.zeros((len(pids_tensor), num_timesteps))
    
        # create padded array X and masked array
        for pid_ind in (tqdm(range(len(pids)))):
            pid = pids[pid_ind]

            X_tensor[pid_ind, :] = torch.from_numpy(data_X.loc[pid].values)
            
            pid_W = torch.from_numpy(data_W.loc[pid].values)
            W_tensor[pid_ind, iterations[pid_ind, 0]:iterations[pid_ind, 1]+1, :] = pid_W
            
            pid_Z = torch.from_numpy(data_Z.loc[pid].values)
            Z_tensor[pid_ind, iterations[pid_ind, 0]:iterations[pid_ind, 1]+1, :] = pid_Z
            
            X_mask[pid_ind, iterations[pid_ind, 0]:iterations[pid_ind, 1]+1] = 1
                
        # get y array
        y_label = torch.from_numpy(self.df_pop.set_index('person_id').loc[pids, 'sz_flag'].values)
    
        dataset_object = torch.utils.data.TensorDataset(pids_tensor, X_tensor, Z_tensor, W_tensor, X_mask, y_label)
        return dataset_object
