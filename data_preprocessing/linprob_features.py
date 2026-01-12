import pandas as pd 
import numpy as np 
import torch
import glob
import os
import pickle

disease = 'T2DM'
print(disease.lower())

def load_and_filter_parquets(
    folder_path: str,
    cohort_df: pd.DataFrame,
    person_id_col: str = "person_id",
    verbose: bool = True,
):
    """
    Read all parquet files in a folder, filter by person_id and date window,
    and return a concatenated DataFrame.

    Args:
        folder_path: Path to folder containing parquet files
        cohort_df: DataFrame with person_id, observation_start_date, cohort_start_date
        start_date_col: Column name in parquet files to filter by date
        person_id_col: Person ID column name
        obs_start_col: Observation start date column in cohort_df
        cohort_start_col: Cohort start date column in cohort_df
        verbose: Whether to print progress

    Returns:
        pd.DataFrame
    """

    # ---- Fast lookup set ----
    valid_person_ids = set(cohort_df["subject_id"])

    # ---- Index for merge ----
    cohort_df = cohort_df.set_index("subject_id")

    dfs = []
    parquet_files = sorted(glob.glob(os.path.join(folder_path, "*.parquet")))

    if verbose:
        print(f"Found {len(parquet_files)} parquet files")

    for i, path in enumerate(parquet_files):
        if verbose:
            print(f"[{i+1}/{len(parquet_files)}] Reading {os.path.basename(path)}")

        df = pd.read_parquet(path)

        # ---- Filter person_id early ----
        df = df[df[person_id_col].isin(valid_person_ids)]

        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("No data matched filters.")
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)

    if verbose:
        print(f"Final dataframe shape: {result.shape}")

    return result

def merge_demographics(all_pids, person_df):
    # get simplified race_concept_id
    person_df['simplified_race_concept_id'] = person_df['race_concept_id'].copy()
    person_df.loc[person_df['simplified_race_concept_id']==8552, 'simplified_race_concept_id'] = 0
    person_df.loc[person_df['simplified_race_concept_id'].isin([8557, 8657, 38003613]), 'simplified_race_concept_id'] = 8522
    print(person_df['simplified_race_concept_id'].value_counts())

    person_df[['is_Black', 'is_White', 'is_MissingRace', 'is_OtherRace', 'is_Asian', 'is_Male', 'is_Female']] = 0
    person_df.loc[person_df['gender_concept_id']==8507, 'is_Male'] = 1
    person_df.loc[person_df['gender_concept_id']==8532, 'is_Female'] = 1

    person_df.loc[person_df['simplified_race_concept_id']==8527, 'is_White'] = 1
    person_df.loc[person_df['simplified_race_concept_id']==0, 'is_MissingRace'] = 1
    person_df.loc[person_df['simplified_race_concept_id']==8516, 'is_Black'] = 1
    person_df.loc[person_df['simplified_race_concept_id']==8515, 'Asian'] = 1
    person_df.loc[person_df['simplified_race_concept_id']==8522, 'is_OtherRace'] = 1
    person_df = person_df[['person_id', 'race_concept_id', 'simplified_race_concept_id', 'gender_concept_id', 'is_Black', 'is_White', 'is_MissingRace', 'is_OtherRace', 'is_Asian', 'is_Male', 'is_Female']]
    all_pids = all_pids.merge(person_df, left_on = 'subject_id', right_on = 'person_id', how = 'inner')
    return all_pids


def get_unique_dates(root_path, filetype, start_date_colname, all_pids):
    # get prediction-time-specific hcu

    filepath = os.listdir(f'{root_path}/{filetype}/')
    list_files = [i for i in filepath if i.endswith('.parquet')]
    list_dfs = []
    for f in list_files:
        temp_file = pd.read_parquet(f'{root_path}/{filetype}/{f}')
        temp_file = temp_file.loc[temp_file['person_id'].isin(all_pids['subject_id'])]
        temp_file = temp_file[['person_id', start_date_colname]].drop_duplicates()
        temp_file = temp_file.merge(all_pids, how='inner', on = 'person_id')
        temp_file[start_date_colname] = pd.to_datetime(temp_file[start_date_colname])
        
        temp_file = temp_file.loc[temp_file[start_date_colname] >= temp_file['observation_start_time']]
        temp_file = temp_file.loc[temp_file[start_date_colname] <= temp_file['prediction_time']]
        temp_file = temp_file[['PID_unique', start_date_colname]].drop_duplicates()
        
        list_dfs.append(temp_file)
    final_df = pd.concat(list_dfs)
    final_df.rename({start_date_colname: 'start_date'}, axis=1, inplace=True)
    final_df = final_df.drop_duplicates()
    print(f'Finished {filetype}: {len(final_df)}')
    return final_df

root_path = 'PATH'
phenotypes_path = 'PATH'
llama_path = 'PATH'
save_path = f'PATH'

# get patient_id, prediction_time, and labels
train_pids = pd.read_parquet(f'{phenotypes_path}/{disease}/train.parquet')
val_pids = pd.read_parquet(f'{phenotypes_path}/{disease}/tuning.parquet')
test_pids = pd.read_parquet(f'{phenotypes_path}/{disease}/held_out.parquet')

all_pids = pd.concat([train_pids, val_pids, test_pids])
all_pids['prediction_time'] = pd.to_datetime(all_pids['prediction_time'])
person_df = load_and_filter_parquets(f'{root_path}/person', all_pids, verbose = False)
print(len(all_pids))
all_pids = merge_demographics(all_pids, person_df)
all_pids['observation_start_time'] = all_pids['prediction_time'] - pd.Timedelta(days=730)
print(len(all_pids))

arr_pids = np.linspace(1, len(all_pids)+1, len(all_pids), dtype=int)
np.random.shuffle(arr_pids)
all_pids['PID_unique'] = arr_pids

# get hcu time-specific
distinct_conds_dates = get_unique_dates(root_path, 'condition_occurrence', 'condition_start_date', all_pids)
distinct_visit_dates = get_unique_dates(root_path, 'visit_occurrence', 'visit_start_date', all_pids)
distinct_procedure_dates = get_unique_dates(root_path, 'procedure_occurrence', 'procedure_date', all_pids)
distinct_dates = pd.concat([distinct_conds_dates, distinct_visit_dates, distinct_procedure_dates]).drop_duplicates()
distinct_dates = pd.DataFrame(distinct_dates.groupby('PID_unique').count()).reset_index()
print(distinct_dates)
distinct_dates['hcu'] = distinct_dates['start_date']/2 # 2 years of observation

print(len(all_pids))
all_pids = all_pids.merge(distinct_dates[['PID_unique', 'hcu']], on = 'PID_unique', how = 'inner')
print(len(all_pids))

train_df = pd.read_parquet(f'{llama_path}/{disease}/features_with_label/train.parquet')
test_df = pd.read_parquet(f'{llama_path}/{disease}/features_with_label/test.parquet')
full_llama_df = pd.concat([train_df, test_df])

features_df = pd.DataFrame(full_llama_df["features"].to_list(), index=full_llama_df.index)
features_df.columns = [f'embedding_vec_{i}' for i in np.arange(1, features_df.shape[1]+1)]
full_llama_df = pd.concat([full_llama_df.drop(columns="features"), features_df], axis=1)

all_features = full_llama_df.merge(all_pids, on = ['subject_id', 'prediction_time', 'boolean_value'])
print(full_llama_df.shape)
print(all_pids.shape)
print(all_features.shape)
print(all_features.head())
all_features.to_csv(f'SAVE FILE')

