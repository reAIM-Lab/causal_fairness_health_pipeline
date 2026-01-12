import os
import numpy as np
import pickle as pk
from estimation import *
import pandas as pd
import cupy as cp
import pickle 
import time
from sklearn.metrics import *


def run_cfa_output(model_name, model_path_base, results_path_base, df_data, featureset, features):
    train_outputs = pd.read_csv(f'{model_path_base}/train_outputs.csv')
    val_outputs = pd.read_csv(f'{model_path_base}/val_outputs.csv')
    test_outputs = pd.read_csv(f'{model_path_base}/test_outputs.csv')

    all_outputs = pd.concat([train_outputs, val_outputs, test_outputs])
    def get_cutoff_prob(y_true, y_pred):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        idx = np.argmax(tpr - fpr)
        cutoff_prob = thresholds[idx]
        print(cutoff_prob)
        return cutoff_prob
    cutoff_prob = get_cutoff_prob(val_outputs['y_true'], val_outputs['y_pred'])
    all_outputs['y_out'] = (all_outputs['y_pred'] > cutoff_prob)*1
    print(f'checking length of dataframe: {len(df_data), len(all_outputs)}')
    df_data = df_data.merge(all_outputs[['person_id', 'pred_time', 'y_out']], how = 'inner', left_on = ['person_id', 'prediction_time'], right_on = ['person_id', 'pred_time'])
    print(f'checking length of dataframe: {len(df_data)}')

    Z_cols = features['Z']
    W_cols = features['W']
    print(len(Z_cols), len(W_cols))
    dict_dims = {'X': 1, 'Y': 1, 'W': len(W_cols), 'Z': len(Z_cols)}
    model_path = f'{model_path_base}/effect_estimation_models/{featureset}'
    os.makedirs(model_path, exist_ok=True)

    # run train_split
    train_df = df_data.loc[df_data['split']=='train']
    train_results_df, train_submodel_performances_df = train_estimate_effects(train_df, X_col, y_col, W_cols, Z_cols, True, 
                                                                            len(df_data), device, clip=1e-4, K=5, 
                                                                            bootstraps=100, sample_kwargs={'dims':dict_dims}, 
                                                                            hparam_path=model_path)

    train_results_df.to_csv(f'{results_path}/estimated_effects_splittrain_features{featureset}.csv')
    train_submodel_performances_df.to_csv(f'{results_path}/submodel_performances_splittrain_features{featureset}.csv')

    # run val split
    val_df = df_data.loc[df_data['split']=='tuning']
    val_results_df, val_submodel_performances_df = test_estimate_effects(val_df, X_col, y_col, W_cols, Z_cols, 
                                                                        True, device, clip=1e-4, K=5, bootstraps=100, model_path=model_path)
    val_results_df.to_csv(f'{results_path}/estimated_effects_splitval_features{featureset}.csv')
    val_submodel_performances_df.to_csv(f'{results_path}/submodel_performances_splitval_features{featureset}.csv')

    # run test split
    test_df = df_data.loc[df_data['split']=='held_out']
    test_results_df, test_submodel_performances_df = test_estimate_effects(test_df, X_col, y_col, W_cols, Z_cols, 
                                                                        True, device, clip=1e-4, K=5, bootstraps=100, model_path=model_path)
    test_results_df.to_csv(f'{results_path}/estimated_effects_splittest_features{featureset}.csv')
    test_submodel_performances_df.to_csv(f'{results_path}/submodel_performances_splittest_features{featureset}.csv')

if __name__ == '__main__':
    # SENSITIVE ATTRIBUTE IS GENDER
    cp.cuda.Device(3).use() 
    device = 'cuda:3'
    featureset = '20percentpfi'
    
    disease = 'AMI' # OR SLE
    print(disease)

    path = f'PATH'
    dataset_prefix = f'12_18_{disease.lower()}_llama_'

    df_data = pd.read_csv(f'{path}/{dataset_prefix}features.csv', index_col = 0)

    # set up columns! 
    all_cols = list(df_data.columns)
    cols_dict = {}

    if featureset == '20percentpfi':
        with open(f'{path}/top_20percent_features_pfi_{disease.lower()}.pkl', "rb") as fp:   #Pickling
            w_cols = pickle.load(fp)
    elif featureset == 'fullfeatures': 
        w_cols = [i for i in all_cols if 'embedding_vec' in i]
    print(w_cols[0:10])
    cols_dict['W'] = w_cols

    cols_dict['Z'] = ['hcu', 'is_Black', 'is_White', 'is_MissingRace', 'is_OtherRace', 'is_Asian']

    X_col = 'is_Male'
    y_col = 'y_out'
    clip = 1e-4

    # Call this across baseline model and all fairness intervention
    model_name = 'NAME OF MODEL + INTERVENTION'
    model_path_base = f'PATH TO MODEL'
    results_path = f'PATH TO RESULTS' 
    os.makedirs(results_path, exist_ok=True)
    run_cfa_output(model_name, model_path_base, results_path, df_data, featureset, cols_dict) 

    # SENSITIVE ATTRIBUTE IS RACE
    disease = 'T2DM'
    cp.cuda.Device(3).use() 
    device = 'cuda:3'
    featureset = '20percentpfi'

    path = f'PATH'
    dataset_prefix = f'12_18_{disease.lower()}_llama_'

    df_data = pd.read_csv(f'{path}/{dataset_prefix}features.csv', index_col = 0)

    # set up columns! 
    all_cols = list(df_data.columns)
    cols_dict = {}
    if featureset == '20percentpfi':
        with open(f'{path}/top_20percent_features_pfi_{disease.lower()}.pkl', "rb") as fp:   #Pickling
            w_cols = pickle.load(fp)
    elif featureset == 'fullfeatures': 
        w_cols = [i for i in all_cols if 'embedding_vec' in i]
    print(w_cols[0:10])
    cols_dict['W'] = w_cols

    #cols_dict['Z'] = ['hcu', 'is_Black', 'is_White', 'is_MissingRace', 'is_OtherRace', 'is_Asian']
    cols_dict['Z'] = ['hcu', 'is_Male']

    X_col = 'is_Black'
    y_col = 'y_out'
    clip = 1e-4
    print(len(df_data))
    df_data = df_data.loc[(df_data['is_Black']==1) | (df_data['is_White']==1)]
    print(len(df_data))

    # Call this across baseline model and all fairness interventions
    model_name = 'NAME OF MODEL + INTERVENTION'
    model_path_base = f'PATH TO MODEL'
    results_path = f'PATH TO RESULTS' 
    os.makedirs(results_path, exist_ok=True)
    run_cfa_output(model_name, model_path_base, results_path, df_data, featureset, cols_dict) 



    