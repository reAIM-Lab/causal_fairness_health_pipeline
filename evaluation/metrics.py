import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.utils import resample

def bootstrap_metric(y_true, y_pred, metric_func, n_bootstraps=300, ci=95):
    """
    Returns mean, lower_ci, upper_ci for a given metric using bootstrapping.
    """
    stats = []
    # Seed for reproducibility
    rng = np.random.RandomState(31)
    
    for _ in range(n_bootstraps):
        # Stratified resampling is often preferred for classification
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue # Skip if only one class is present
        score = metric_func(y_true[indices], y_pred[indices])
        stats.append(score)
    
    alpha = (100 - ci) / 2.0
    lower = np.percentile(stats, alpha)
    upper = np.percentile(stats, 100 - alpha)
    mean_val = metric_func(y_true, y_pred) # Use the actual mean, not bootstrap mean
    
    return mean_val, lower, upper

def get_performance_row(model_path, split_name):
    """
    Loads model outputs and calculates performance metrics.
    Assumes standard columns 'label' and 'prediction' (probabilities).
    """
    try:
        # Construct path (assuming standard naming convention defined in prompt)
        # You might need to adjust the file naming logic here
        file_path = f"{model_path}/{split_name}_outputs.csv"
        df = pd.read_csv(file_path)
        
        # Ensure correct types
        y_true = df['y_true'].values
        y_pred = df['y_pred'].values
        
        # BCE (No CI required per instructions)
        bce = log_loss(y_true, y_pred)
        
        # Bootstrapped Metrics
        auroc_mean, auroc_low, auroc_high = bootstrap_metric(y_true, y_pred, roc_auc_score)
        brier_mean, brier_low, brier_high = bootstrap_metric(y_true, y_pred, brier_score_loss)
        
        return {
            f'{split_name}_bce': bce,
            f'{split_name}_auroc_mean': auroc_mean,
            f'{split_name}_auroc_ci_low': auroc_low,
            f'{split_name}_auroc_ci_high': auroc_high,
            f'{split_name}_calibration_mean': brier_mean,
            f'{split_name}_calibration_ci_low': brier_low,
            f'{split_name}_calibration_ci_high': brier_high,
        }
    except FileNotFoundError:
        print(f"Warning: Could not find {file_path}")
        return {}