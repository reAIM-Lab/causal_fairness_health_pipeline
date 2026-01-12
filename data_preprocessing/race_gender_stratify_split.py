import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

path = 'PATH'
raw_data_path = f'PATH'
causal_path = f'PATH'

df_pop = pd.read_csv('READ IN DF_POP')

# CREATE A NEW STRATIFICATION THAT USES RACE, GENDER AND OUTCOME AND DO THE ABOVE AGAIN
pid_trainval, pid_test = train_test_split(df_pop['person_id'], stratify=df_pop[['race_concept_id', 'gender_concept_id', 'sz_flag']], test_size=0.2, random_state = 4)
trainval_pop = df_pop.loc[df_pop['person_id'].isin(pid_trainval)]
pid_train, pid_val = train_test_split(trainval_pop['person_id'], stratify=trainval_pop['sz_flag'], test_size=1/8, random_state = 4)
print(len(pid_train), len(pid_val), len(pid_test))
print(len(pid_train)/len(df_pop), len(pid_val)/len(df_pop), len(pid_test)/len(df_pop))

df_split_stratified = df_pop[['person_id', 'sz_flag', 'race_concept_id']]
df_split_stratified['split'] = 'train'
df_split_stratified.loc[df_split_stratified['person_id'].isin(pid_val), 'split'] = 'val'
df_split_stratified.loc[df_split_stratified['person_id'].isin(pid_test), 'split'] = 'test'
df_split_stratified.to_csv(f'{causal_path}/tvt_split_stratified_2dx.csv')
