import os
import numpy as np
import pickle as pk
from estimation import *
import pandas as pd
import cupy as cp
import pickle 
import time
from sklearn.metrics import *

"""
SET DISEASE AND PATH, LOAD IN DATA


# if T2DM or SCZ, constrict to black vs. white
df_pop = pd.read_csv(f'{path}/population.csv')
df_pop = df_pop.loc[df_pop['race_concept_id'].isin([8516, 8527])]
X_col = 'is_Black' # SPECIFY
y_col = 'boolean_value'

"""

dict_features = {'top20percent': data_columns} 

df_split = pd.read_csv(f'{path}/tvt_split_stratified.csv', index_col=0)


# create demographic awareness
df_pop[['is_White', 'is_Black', 'is_MissingRace', 'is_Male', 'is_Female']] = 0
df_pop.loc[df_pop['race_concept_id']==8527, 'is_White'] = 1
df_pop.loc[df_pop['race_concept_id']==8516, 'is_Black'] = 1
df_pop.loc[df_pop['race_concept_id']==0, 'is_MissingRace'] = 1
df_pop.loc[df_pop['gender_concept_id']==8507, 'is_Male'] = 1
df_pop.loc[df_pop['gender_concept_id']==8532, 'is_Female'] = 1



# merge with causal data

clip = 1e-4
for featureset in dict_features.keys():
    features = dict_features[featureset]
    Z_cols = features['Z']
    W_cols = features['W']
    print(len(Z_cols), len(W_cols))
    dict_dims = {'X': 1, 'Y': 1, 'W': len(W_cols), 'Z': len(Z_cols)}
    model_path = f'{model_path_base}/{featureset}'
    os.makedirs(model_path, exist_ok=True)
    # run train_split
    train_df = df_data.loc[df_data['split']=='train']
    train_results_df, train_submodel_performances_df = train_estimate_effects(train_df, X_col, y_col, W_cols, Z_cols, True, 
                                                                              len(df_data), device, clip=clip, K=5, 
                                                                              bootstraps=100, sample_kwargs={'dims':dict_dims}, 
                                                                              hparam_path=model_path)

    train_results_df.to_csv(f'{results_path}/estimated_effects_splittrain_features{featureset}.csv')
    train_submodel_performances_df.to_csv(f'{results_path}/submodel_performances_splittrain_features{featureset}.csv')
    
    # run val split
    val_df = df_data.loc[df_data['split']=='val']
    val_results_df, val_submodel_performances_df = test_estimate_effects(val_df, X_col, y_col, W_cols, Z_cols, 
                                                                        True, device, clip=clip, K=5, bootstraps=100, model_path=model_path)
    val_results_df.to_csv(f'{results_path}/estimated_effects_splitval_features{featureset}.csv')
    val_submodel_performances_df.to_csv(f'{results_path}/submodel_performances_splitval_features{featureset}.csv')

    # run test split
    test_df = df_data.loc[df_data['split']=='test']
    test_results_df, test_submodel_performances_df = test_estimate_effects(test_df, X_col, y_col, W_cols, Z_cols, 
                                                                        True, device, clip=clip, K=5, bootstraps=100, model_path=model_path)
    test_results_df.to_csv(f'{results_path}/estimated_effects_splittest_features{featureset}.csv')
    test_submodel_performances_df.to_csv(f'{results_path}/submodel_performances_splittest_features{featureset}.csv')


