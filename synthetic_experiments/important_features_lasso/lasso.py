import os
import numpy as np
import pickle as pk
import pandas as pd
import cupy as cp
import sys
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

sys.path.append('../')
from estimation import *

def run_lasso(df_full_features, all_cols, Y_col, target_num_features, max_iter=10000, tol=1e-4):    
    """
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
    selected_features : list of str
        Feature names selected by Lasso (nonzero coefficients).
    lasso_model : sklearn.linear_model.Lasso
        The fitted Lasso model.
    """

    X = df_full_features[all_cols].copy()
    y = df_full_features[Y_col].copy()

    # Standardize predictors (important for Lasso)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Search for alpha giving ~target_num_features
    alphas = np.logspace(-4, 1, 100)
    selected_features = []
    lasso_model = None

    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=max_iter, tol=tol, random_state=42)
        lasso.fit(X_scaled, y)
        nonzero_features = np.sum(lasso.coef_ != 0)

        if nonzero_features <= target_num_features:
            selected_features = list(np.array(all_cols)[lasso.coef_ != 0])
            lasso_model = lasso
            break

    # If alpha never reduces enough features, pick smallest alpha
    if not selected_features:
        lasso = Lasso(alpha=alphas[-1], max_iter=max_iter, tol=tol, random_state=42)
        lasso.fit(X_scaled, y)
        selected_features = list(np.array(all_cols)[lasso.coef_ != 0])
        lasso_model = lasso

    print(f"Selected {len(selected_features)} features (target={target_num_features}).")
    return selected_features, lasso_model
