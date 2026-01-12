import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import os
import pickle
import itertools
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Ensure utils are in path (matching your directory structure)
sys.path.append('../nontemporal_model_training/utils')
import models

# --- configuration ---
SEED_VALUE = 35
TOP_PERCENT = 0.2  # Keep top 20% of features
N_REPEATS = 100     # Number of permutations per feature

torch.manual_seed(SEED_VALUE)
np.random.seed(SEED_VALUE)

def permutation_feature_importance_torch(model, X_tensor, feature_names, device, n_repeats=N_REPEATS):
    """
    Computes PFI for a PyTorch model, but ONLY for features starting with 'embedding_vec'.
    """
    model.eval()
    X_device = X_tensor.to(device)
    n_samples, n_features = X_device.shape
    
    # 1. Get Base Predictions
    with torch.no_grad():
        base_preds = torch.sigmoid(model(X_device))
    
    importances = {}
    
    print("Calculating Permutation Feature Importance (Embedding Vecs Only)...")
    
    # We iterate through ALL features to keep indices (i) aligned with the tensor columns
    for i, col_name in tqdm(enumerate(feature_names), total=len(feature_names)):
        
        # --- FILTER: Skip if not an embedding vector ---
        if not col_name.startswith("embedding_vec"):
            continue

        # Save original column
        original_col = X_device[:, i].clone()
        
        diffs = []
        for _ in range(n_repeats):
            # Shuffle column
            perm_indices = torch.randperm(n_samples, device=device)
            X_device[:, i] = original_col[perm_indices]
            
            # Predict
            with torch.no_grad():
                perm_preds = torch.sigmoid(model(X_device))
            
            # Calc difference
            diff = torch.abs(base_preds - perm_preds).mean().item()
            diffs.append(diff)
        
        # Restore original column
        X_device[:, i] = original_col
        
        # Store result
        importances[col_name] = np.mean(diffs)

    # Convert to DataFrame
    importance_df = (
        pd.DataFrame(list(importances.items()), columns=["feature", "importance"])
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print(len(importance_df))
    return importance_df

def run_pfi_pipeline(disease, model_dir, top_n_percent=0.2):
    print(f'Running PFI for: {disease} - {awareness}')
    
    # 1. Setup Paths
    data_path = f'PATH'
    
    device = torch.device("cuda:3")
    
    # 2. Load Data
    print("Loading data...")
    data_df = pd.read_csv(f'{data_path}/12_18_{disease}_llama_features.csv', index_col=0)

    # 3. Feature Selection & Scaling (Exact replication of training logic)
    all_columns = list(data_df.columns)
    embedding_cols = [i for i in all_columns if 'embedding_vec' in i]
    demo_cols = [i for i in all_columns if 'is_' in i]
    hcu_col = ['hcu']
    
    data_columns = embedding_cols + hcu_col + demo_cols        
    data_df[data_columns] = data_df[data_columns].astype('float32')

    # Fit Scaler on TRAIN, Transform on ALL (we will use Test/Held-out for PFI)
    scaler = StandardScaler()
    train_mask = data_df['split'] == 'train'
    scaler.fit(data_df.loc[train_mask, data_columns])
    data_df.loc[:, data_columns] = scaler.transform(data_df[data_columns])
    
    # Extract Test Set (Standard practice: evaluate importance on held-out data)
    test_mask = data_df['split'] == 'held_out'
    X_test = data_df.loc[test_mask, data_columns].values
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # 4. Load Model
    print("Loading model...")
    # Load config to instantiate the class structure
    try:
        model_config = torch.load(f"{model_dir}/best_model_config.pt", weights_only=False)
        # Instantiate model
        model = model_config['name'](**model_config['params']).to(device)
        # Load weights
        model.load_state_dict(torch.load(f"{model_dir}/best_model.pt", map_location=device, weights_only=False))
    except FileNotFoundError:
        print(f"Model files not found in {model_dir}. Skipping.")
        return

    # 5. Run Permutation Feature Importance
    importance_df = permutation_feature_importance_torch(
        model=model,
        X_tensor=X_test_tensor,
        feature_names=data_columns,
        device=device,
        n_repeats=N_REPEATS
    )

    # 6. Filter Top N%
    num_keep = int(len(importance_df) * top_n_percent)
    if num_keep < 1: num_keep = 1 # Ensure at least 1 feature is kept
    
    top_features_df = importance_df.head(num_keep)
    print(top_features_df.head())
    top_features_list = top_features_df['feature'].tolist()

    print(f"Top {len(top_features_list)} features ({top_n_percent*100}%) selected.")

    # 7. Save Results
    # Save full importance dataframe (CSV)
    importance_df.to_csv(os.path.join(data_path, "feature_importance_pfi_top20.csv"), index=False)
    
    # Save top features list (Pickle)
    save_path_pkl = os.path.join(data_path, f"top_{int(top_n_percent*100)}percent_features_pfi_{disease.lower()}.pkl")
    with open(save_path_pkl, 'wb') as f:
        pickle.dump(top_features_list, f)
        
    print(f"Saved results to {data_path}")
    print("-" * 50)

# --- Execution ---
if __name__ == "__main__":
    # Pairs to run
    tasks = [
        ('ami', 'PATH'),
        ('sle', 'PATH'),
        ('t2dm', 'PATH')
    ]

    for disease, awareness in tasks:
        run_pfi_pipeline(disease, awareness, top_n_percent=TOP_PERCENT)