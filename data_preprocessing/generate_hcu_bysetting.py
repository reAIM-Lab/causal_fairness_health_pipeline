import numpy as np
import os
import pandas as pd
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from collections import Counter
from datetime import datetime
import sys
import gc
import pickle 
import math

from preprocessing_utils import *

"""
PATHS
"""

# how much data we are using to make predictions
forward_iterations = 5 # 1 year (5 so we end up with 4 iterations)
backwards_iterations = 65 
days_per_iter = 90 # interval size

# censor date to cohort start date
num_days_prediction = 90

# read in population dataframe
df_pop = pd.read_csv(f'{data_path}/population_2dx.csv', parse_dates = ['psychosis_diagnosis_date', 'scz_diagnosis_date', 'cohort_start_date'])
print(len(df_pop), df_pop['sz_flag'].sum()/len(df_pop), len(df_pop['person_id'].unique()))
# instead of num_days_prediction, we want to only look for people who have a cohort start date 1 yr + num_days_prediction after start date
df_pop = df_pop.loc[(df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days >= num_days_prediction + 365]
print(len(df_pop), df_pop['sz_flag'].sum()/len(df_pop), len(df_pop['person_id'].unique()))
df_pop['censor_date'] = df_pop['cohort_start_date'] - pd.Timedelta(days=num_days_prediction)

count_visits = pd.read_csv(f'{int_path}/hcu_visit_counts.csv', parse_dates = ['first_visit'])
df_pop = df_pop.merge(count_visits[['person_id', 'first_visit']], how = 'left', on = 'person_id')

all_visits = pd.read_csv(f'{data_path}/temporal_visits.csv', parse_dates = ['cohort_start_date', 'visit_start_date', 'visit_end_date'])
all_visits = pre_censor_data(all_visits, df_pop, 'visit_start_date')
all_visits.loc[all_visits['visit_end_date'] > all_visits['censor_date'], 'visit_end_date'] = all_visits.loc[all_visits['visit_end_date'] > all_visits['censor_date'], 'censor_date']
print('Duplicate Visits (should be True)', 'Unnamed: 0' not in all_visits.columns, len(all_visits) == len(all_visits['visit_occurrence_id'].unique()))

# get the cutoff date
df_pop = df_pop[['person_id', 'gender_concept_id', 'first_visit', 'cohort_start_date', 'psychosis_diagnosis_date', 'censor_date']]
df_pop['cutoff_pred_date'] = df_pop['psychosis_diagnosis_date']+pd.Timedelta(5*days_per_iter, 'days')
df_pop.loc[df_pop['cutoff_pred_date'] > df_pop['censor_date'], 'cutoff_pred_date'] = df_pop.loc[df_pop['cutoff_pred_date'] > df_pop['censor_date'], 'censor_date']

# restrict to all visits before this cutoff date
all_visits = all_visits.merge(df_pop[['person_id', 'cutoff_pred_date']])
all_visits = all_visits.loc[all_visits['visit_start_date'] <= all_visits['cutoff_pred_date']]

all_visits = all_visits[['person_id', 'visit_start_date']].drop_duplicates()
count_visits = pd.DataFrame(all_visits.groupby('person_id').count())

# merge back with df_pop
df_pop = df_pop.merge(count_visits, how = 'inner', left_on = 'person_id', right_index = True)
df_pop.rename({'visit_start_date': 'num_visits'}, axis='columns', inplace=True)
print(len(df_pop))
print(df_pop.columns)
df_pop['is_Male'] = 0
df_pop.loc[df_pop['gender_concept_id']==8507, 'is_Male'] = 1
df_pop['years_obs'] = (df_pop['cutoff_pred_date'] - df_pop['first_visit']).dt.days
df_pop['hcu'] = df_pop['num_visits']/df_pop['years_obs']

print(df_pop.head())
print(df_pop.isna().sum())
print(df_pop.dtypes)
df_pop.to_csv(f'{int_path}/hcu_perpatient_aggregate.csv')