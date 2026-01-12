import os
import numpy as np
import pickle as pk
import pandas as pd
import cupy as cp
import sys
from tqdm import tqdm

sys.path.append('../')
from estimation import *

def get_most_important_features(df_full_features, all_cols, reduce_cols, Y_col, device, binary = True):    
    """
    Fit XGBoost model and return feature importances
    ----------
    dataframe : pd.DataFrame
        Data containing features and target.
    all_cols : list[str]
        All feature columns used for training.
    reduce_cols : list[str]
        Subset of feature columns to extract importances for.
    Y_col : str
        Target column.
    device : str, default="cpu"
        Device to use ('cpu' or 'cuda').
    binary : bool, default=True
        If True, use XGBClassifier; else XGBRegressor.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'importance'], sorted by importance.
    """
    
    # Extract X, Y
    X = df_full_features[all_cols]
    Y = df_full_features[Y_col]

    # Run grid search
    best_params = grid_search_xgb(X, Y, device, binary, None)

    # Build model with best params
    model_class = xgb.XGBClassifier if binary else xgb.XGBRegressor
    model = model_class(
        tree_method='gpu_hist',
        device=device,
        verbosity=0,
        **best_params
    )
    model.fit(X, Y)
    print('done w model')

    importance_df = permutation_feature_importance(model, X, Y, True)

    return importance_df


def permutation_feature_importance(model, X, Y, binary, n_repeats=5, random_state=31):
    """
    Compute permutation feature importance by shuffling each column and
    measuring change in predictions.
    """
    rng = np.random.default_rng(random_state)
    base_pred = model.predict_proba(X)[:, 1] if binary else model.predict(X)

    importances = {}
    for col in tqdm(X.columns):
        diffs = []
        for _ in range(n_repeats):
            X_permuted = X.copy(deep=True)
            X_permuted[col] = rng.permutation(X_permuted[col].values)
            perm_pred = model.predict_proba(X_permuted)[:, 1] if binary else model.predict(X_permuted)

            # Option 1: mean absolute change in prediction
            diff = np.mean(np.abs(perm_pred - base_pred))
            diffs.append(diff)

        importances[col] = np.mean(diffs)

    importance_df = (
        pd.DataFrame(list(importances.items()), columns=["feature", "importance"])
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
        .set_index("feature")
    )
    return importance_df
