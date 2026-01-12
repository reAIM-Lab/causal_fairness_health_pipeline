import os
import numpy as np
import pickle as pk
import pandas as pd
import cupy as cp
import sys
import scipy.stats as stats

def get_most_unfair_features(df, check_cols, protected_col = 'X'):
    """
    df = data
    check_cols = w_cols (or whichever cols you want to rank)
    protected_col = col which you want to check "unfairness" across
    """
    group0 = df[df[protected_col] == 0]
    group1 = df[df[protected_col] == 1]

    results = []
    for col in check_cols:
        vals0 = group0[col].dropna()
        vals1 = group1[col].dropna()

        if len(vals0) > 1 and len(vals1) > 1:
            stat, pval = stats.ttest_ind(vals1, vals0, equal_var=False)  # Welch’s t-test
            diff_means = vals1.mean() - vals0.mean()
        else:
            diff_means, pval = np.nan, np.nan

        results.append((col, diff_means, pval))

    results_df = pd.DataFrame(results, columns=["feature", "diff_means", "p_value"])
    results_df.set_index("feature", inplace=True)

    # Adjust p-values for multiple comparisons (Bonferroni)
    results_df["adj_p_value"] = results_df["p_value"]*len(check_cols)

    # Sort by adjusted p-value
    results_df = results_df.sort_values(
        by=["adj_p_value", "diff_means"],
        ascending=[True, False]  # adj_p_value ascending, diff_means descending
    )

    return results_df

