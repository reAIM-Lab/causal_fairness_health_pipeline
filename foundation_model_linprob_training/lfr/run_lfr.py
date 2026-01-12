import pandas as pd
import numpy as np
import pickle
import itertools
import os
import torch

from lfr_gpu import LFR_GPU
from sklearn.preprocessing import StandardScaler
from aif360.algorithms.preprocessing import LFR
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.metrics import log_loss, roc_auc_score

np.random.seed(31)
torch.manual_seed(31)
torch.cuda.manual_seed(31)

def main(disease, prot_attr):
    print(f'Running {disease}')
    path = 'PATH TO DATA'
    model_path = 'MODEL PATH'
    os.makedirs(model_path, exist_ok=True)
    device = torch.device("cuda:2")

    # import data
    data_df = pd.read_csv(f'{path}/12_18_{disease}_llama_features.csv', index_col = 0)

    # decide which columns to keep
    all_columns = list(data_df.columns)
    embedding_cols = [i for i in all_columns if 'embedding_vec' in i]
    demo_cols = [i for i in all_columns if 'is_' in i]
    hcu_col = ['hcu']
    data_columns = embedding_cols + demo_cols + hcu_col
    data_columns.remove(prot_attr)

    data_df[data_columns] = data_df[data_columns].astype('float32')

    scaler = StandardScaler()
    train_mask = data_df['split'] == 'train'
    scaler.fit(data_df.loc[train_mask, data_columns])
    data_df.loc[:, data_columns] = scaler.transform(data_df[data_columns])
    print('Done scaling features')

    label_name = "boolean_value"

    df_data_train = data_df.loc[data_df['split']=='train']
    df_data_val = data_df.loc[data_df['split']=='tuning']
    df_data_test = data_df.loc[data_df['split']=='held_out']

    # Helper to create AIF360 datasets
    def create_dataset(df):
        return BinaryLabelDataset(
            df=df[data_columns + [label_name, prot_attr]], 
            label_names=[label_name], 
            protected_attribute_names=[prot_attr]
        )

    dataset_train = create_dataset(df_data_train)
    dataset_val = create_dataset(df_data_val)
    dataset_test = create_dataset(df_data_test)

    priv = [{prot_attr: 1}]
    unpriv = [{prot_attr: 0}]
    # Parameter Grid
    param_grid = {
        'k': [5, 10],            # Number of prototypes
        'Ax': [0.1, 1],       # Feature reconstruction weight
        'Ay': [1.0, 10.0],  # Label prediction weight
        'Az': [1.0, 50.0],       # Fairness constraint weight
        'learning_rate': [1e-4, 5e-4, 1e-5]
    }

    # Selection Hyperparameter
    # How much do we penalize unfairness relative to BCE?
    # Selection Score = BCE + (FAIRNESS_PENALTY * |SPD|)
    FAIRNESS_PENALTY = 0

    best_selection_score = float('inf') # We want to MINIMIZE this score
    best_params = None
    best_lfr_model = None
    best_metrics = {}

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    combinations = np.random.choice(combinations, size=30, replace=False).tolist()


    print(f"Starting Grid Search over {len(combinations)} combinations...")

    # --- 3. Grid Search Execution ---

    for i, params in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] Testing: {params}")
        
        try:
            lfr = LFR_GPU(unprivileged_groups=unpriv,
                    privileged_groups=priv,
                    k=params['k'], 
                    Ax=params['Ax'], 
                    Ay=params['Ay'], 
                    Az=params['Az'],
                    learning_rate=params['learning_rate'],
                    epochs=5000,
                    device = 'cuda:2',
                    verbose=1)
            
            lfr.fit(dataset_train)
            
            # --- Evaluate on Validation Set ---
            
            # 1. Get Predictions (Labels and Probabilities)
            # LFR.predict() returns a dataset with predicted labels
            val_pred_dataset = lfr.predict(dataset_val)
            
            # Extract arrays
            y_true = dataset_val.labels.flatten()
            y_pred_prob = val_pred_dataset.scores.flatten() # Probabilities
            y_pred_label = val_pred_dataset.labels.flatten() # Hard labels (0/1)

            # 2. Calculate Binary Cross Entropy
            # Clip probabilities to avoid log(0)
            epsilon = 1e-15
            y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
            val_bce = log_loss(y_true, y_pred_prob)
            
            # 3. Calculate Fairness (Statistical Parity Difference)
            # We calculate this on the predicted dataset
            metric_val = BinaryLabelDatasetMetric(
                val_pred_dataset, 
                unprivileged_groups=unpriv,
                privileged_groups=priv
            )
            val_spd = metric_val.statistical_parity_difference()
            
            # 4. Composite Selection Score
            # We want low BCE and low absolute Bias (SPD close to 0)
            selection_score = val_bce + (FAIRNESS_PENALTY * abs(val_spd))
            
            print(f"   BCE: {val_bce:.4f} | SPD: {val_spd:.4f} | Score: {selection_score:.4f}")
            
            # 5. Update Best Model
            if selection_score < best_selection_score:
                best_selection_score = selection_score
                best_params = params
                best_lfr_model = lfr
                best_metrics = {'bce': val_bce, 'spd': val_spd, 'auc': roc_auc_score(y_true, y_pred_prob)}
                print(f"   >>> New Best Model Found!")

        except Exception as e:
            print(f"   Failed: {e}")

    print(f"Best Params: {best_params}")
    print(f"Best Metrics: BCE={best_metrics.get('bce'):.4f}, SPD={best_metrics.get('spd'):.4f}, AUC={best_metrics.get('auc'):.4f}")
    print("="*40 + "\n")

    # --- 4. Saving Results (Features + Predictions) ---

    def save_transformed_results(model, dataset, original_df, filename):
        """
        Transforms data to latent space, adds y_true/y_pred, and saves to CSV.
        """
        # 1. Transform features (Get the latent representations)
        dataset_trans = model.transform(dataset)
        
        # 2. Get predictions (Get probability scores)
        dataset_pred = model.predict(dataset)
        
        # 3. Convert latent features to DataFrame
        df_features, _ = dataset_trans.convert_to_dataframe()
        
        # Identify which columns are the new latent features
        # LFR typically names them logically, but we rely on the returned DF structure
        # We drop the original metadata columns temporarily to isolate features if needed,
        # or just keep the whole transformed set.
        
        # 4. Add key metadata
        # Align by index (AIF360 preserves order)
        df_features['PID_unique'] = original_df['PID_unique'].values
        df_features['person_id'] = original_df['person_id'].values
        df_features['prediction_time'] = original_df['prediction_time'].values
        df_features['y_true'] = dataset.labels.flatten()
        df_features['y_pred_hard'] = dataset_pred.labels.flatten() # Hard prediction
        df_features['y_pred'] = dataset_pred.scores.flatten() # Probability
        
        # Reorder columns to put person_id and labels first for readability
        cols = ['PID_unique', 'person_id', 'prediction_time', 'y_true', 'y_pred_hard', 'y_pred']
        feature_cols = [c for c in df_features.columns if c not in cols]
        df_final = df_features[cols + feature_cols]
        
        save_path = f'{model_path}/{filename}'
        df_final.to_csv(save_path, index=False)
        print(f"Saved processed data to: {save_path}")

    # Save all splits
    save_transformed_results(best_lfr_model, dataset_train, df_data_train, f"train_outputs.csv")
    save_transformed_results(best_lfr_model, dataset_val, df_data_val, f"val_outputs.csv")
    save_transformed_results(best_lfr_model, dataset_test, df_data_test, f"test_outputs.csv")

    # --- 5. Save Model ---
    model_path_full = f'{model_path}/best_lfr_model.pkl'
    with open(model_path_full, 'wb') as f:
        pickle.dump(best_lfr_model, f)
    print(f"Model saved to {model_path_full}")

main('ami', 'is_Male')
main('sle', 'is_Male')
main('t2dm', 'is_Black')
