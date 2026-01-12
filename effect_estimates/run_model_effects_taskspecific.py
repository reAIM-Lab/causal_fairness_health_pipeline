import os
import numpy as np
import pickle as pk
from estimation import *
import pandas as pd
import cupy as cp
import pickle 
import time
from sklearn.metrics import *

def run_cfa_output(model_name, model_path_base, results_path_base, df_pop):
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
    df_pop = df_pop.merge(all_outputs[['person_id', 'y_out']], how = 'inner', on = 'person_id')

    # merge with causal data
    df_data = pd.read_csv(f'{int_path}/{dataset_prefix}causal_data.csv')
    df_data = df_data.merge(df_pop[['person_id', 'y_out', 'is_White', 'is_Black', 'is_MissingRace', 'is_Male', 'is_Female']], how = 'inner', on = 'person_id')
    df_data = df_data.merge(df_split[['person_id', 'split']], how = 'inner', on = 'person_id')

    for featureset in dict_features.keys():
        features = dict_features[featureset]
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
        val_df = df_data.loc[df_data['split']=='val']
        val_results_df, val_submodel_performances_df = test_estimate_effects(val_df, X_col, y_col, W_cols, Z_cols, 
                                                                            True, device, clip=1e-4, K=5, bootstraps=100, model_path=model_path)
        val_results_df.to_csv(f'{results_path}/estimated_effects_splitval_features{featureset}.csv')
        val_submodel_performances_df.to_csv(f'{results_path}/submodel_performances_splitval_features{featureset}.csv')

        # run test split
        test_df = df_data.loc[df_data['split']=='test']
        test_results_df, test_submodel_performances_df = test_estimate_effects(test_df, X_col, y_col, W_cols, Z_cols, 
                                                                            True, device, clip=1e-4, K=5, bootstraps=100, model_path=model_path)
        test_results_df.to_csv(f'{results_path}/estimated_effects_splittest_features{featureset}.csv')
        test_submodel_performances_df.to_csv(f'{results_path}/submodel_performances_splittest_features{featureset}.csv')

if __name__ == '__main__':
    """
    PATHS
    """
    with open(f'{int_path}/{dataset_prefix}colnames_dict_xrace', "rb") as fp:   #Pickling
        data_columns = pickle.load(fp)
    with open(f'{int_path}/top20percent_w_10_31_mdcd_2dx_1yrpred_colnames', "rb") as fp:   #Pickling
        data_columns_w = pickle.load(fp)
    data_columns['W'] = data_columns_w

    df_split = pd.read_csv(f'{int_path}/tvt_split_stratified_2dx.csv', index_col=0)

    # constrict to black vs. white
    df_pop = pd.read_csv(f'{data_path}/population_2dx.csv')
    df_pop = df_pop.loc[df_pop['race_concept_id'].isin([8516, 8527])]
    # create demographic awareness
    df_pop[['is_White', 'is_Black', 'is_MissingRace', 'is_Male', 'is_Female']] = 0
    df_pop.loc[df_pop['race_concept_id']==8527, 'is_White'] = 1
    df_pop.loc[df_pop['race_concept_id']==8516, 'is_Black'] = 1
    df_pop.loc[df_pop['race_concept_id']==0, 'is_MissingRace'] = 1
    df_pop.loc[df_pop['gender_concept_id']==8507, 'is_Male'] = 1
    df_pop.loc[df_pop['gender_concept_id']==8532, 'is_Female'] = 1

    # CHECK THESE
    cp.cuda.Device(0).use() 
    device = 'cuda:0'
    
    dict_features = {'top20percent': data_columns} 
    X_col = 'is_Black'
    y_col = 'y_out'
    
    # Call this across baseline model and all fairness interventions
    model_name = 'NAME OF MODEL + INTERVENTION'
    model_path_base = f'PATH TO MODEL'
    results_path = f'PATH TO RESULTS' 
    os.makedirs(results_path, exist_ok=True)
    run_cfa_output(model_name, model_path_base, results_path, df_data, featureset, cols_dict) 

    

        
