import pandas as pd
import numpy as np

def calculate_stats(series):
    """
    Helper to calculate mean and 95% CI from a series of bootstrap samples.
    """
    if series.empty:
        return np.nan, np.nan, np.nan
    
    mean_val = series.mean()
    # 95% CI corresponds to 2.5th and 97.5th percentiles
    ci_low = np.percentile(series, 2.5)
    ci_high = np.percentile(series, 97.5)
    
    return mean_val, ci_low, ci_high

def summarize_causal_effects(df, keep_cols_effects):
    """
    Calculates statistics (Mean, 95% CI) from raw bootstrap samples for each split,
    then aggregates them into the final row structure.
    """
    row_data = {}
    
    # We define the splits we care about finding max/min over
    splits_to_check = ['train', 'validation', 'test']

    for col in keep_cols_effects:
        # We will track these to find the global min/max across all splits
        split_means = []
        split_lows = []
        split_highs = []

        # 1. Calculate stats for each split independently
        for split in splits_to_check:
            # Filter rows belonging to this split
            split_data = df[df['split'] == split][col]
            
            mean_val, low_val, high_val = calculate_stats(split_data)
            
            # Save specific test/val metrics as required by your schema
            if split == 'test':
                row_data[f'test_{col}_mean'] = mean_val
                row_data[f'test_{col}_ci_low'] = low_val
                row_data[f'test_{col}_ci_high'] = high_val
            elif split == 'validation':
                row_data[f'val_{col}_mean'] = mean_val
                # We don't explicitly need val CI columns in the final CSV per prompt, 
                # but we need them for the global min/max calculation below.

            # Accumulate for global stats if data existed
            if not pd.isna(mean_val):
                split_means.append(mean_val)
                split_lows.append(low_val)
                split_highs.append(high_val)

        # 2. Aggregates across all splits (Min/Max CIs)
        # "widest possible CI" logic: min of all lower bounds, max of all upper bounds
        if split_means:
            row_data[f'mean_{col}_mean'] = np.mean(split_means)
            row_data[f'min_{col}_ci_low'] = np.min(split_lows)
            row_data[f'max_{col}_ci_high'] = np.max(split_highs)
        else:
            row_data[f'mean_{col}_mean'] = np.nan
            row_data[f'min_{col}_ci_low'] = np.nan
            row_data[f'max_{col}_ci_high'] = np.nan
        
    return row_data

def select_best_model(df):
    """
    Selects best model per experiment based on Val BCE and absolute Causal bias.
    Metric: Val BCE + Mean(|NDE|, |NIE|, |SE|)
    """
    selected_rows = []
    
    for exp, group in df.groupby('experiment_name'):
        if len(group) == 1:
            selected_rows.append(group.iloc[0])
            continue
            
        try:
            # Calculate mean absolute bias using the VAL means we calculated above
            causal_bias = (
                group['val_Estimated_NDE_sn_mean'].abs() + 
                group['val_Estimated_NIE_sn_mean'].abs() + 
                group['val_Estimated_SE_sn_mean'].abs()
            ) / 3.0
            
            # Combine with BCE
            selection_score = group['val_bce'] + 3*causal_bias
            
            best_idx = selection_score.idxmin()
            selected_rows.append(group.loc[best_idx])
        except KeyError as e:
            print(f"Skipping selection for {exp} due to missing columns: {e}")
            selected_rows.append(group.iloc[0])

    return pd.DataFrame(selected_rows)

def format_latex_value(mean, low, high, decimals=3):
    if pd.isna(mean):
        return "-"
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} ({fmt.format(low)}, {fmt.format(high)})"

def generate_latex_table(df, output_path):
    latex_rows = []
    
    for _, row in df.iterrows():
        tex_row = {
            'Experiment': row['experiment_name'].replace('_', '\\_'),
            
            # Performance 
            'AUROC': format_latex_value(
                row['test_auroc_mean'], 
                row['test_auroc_ci_low'], 
                row['test_auroc_ci_high']
            ),
            'Calibration': format_latex_value(
                row['test_calibration_mean'], 
                row['test_calibration_ci_low'], 
                row['test_calibration_ci_high']
            ),
            
            # Causal Effects
            'TE': format_latex_value(
                row['test_Estimated_TE_sn_mean'], 
                row['min_Estimated_TE_sn_ci_low'], 
                row['max_Estimated_TE_sn_ci_high']
            ),
            'NDE': format_latex_value(
                row['test_Estimated_NDE_sn_mean'], 
                row['min_Estimated_NDE_sn_ci_low'], 
                row['max_Estimated_NDE_sn_ci_high']
            ),
            'NIE': format_latex_value(
                row['test_Estimated_NIE_sn_mean'], 
                row['min_Estimated_NIE_sn_ci_low'], 
                row['max_Estimated_NIE_sn_ci_high']
            ),
            'SE': format_latex_value(
                row['test_Estimated_SE_sn_mean'], 
                row['min_Estimated_SE_sn_ci_low'], 
                row['max_Estimated_SE_sn_ci_high']
            ),
        }
        latex_rows.append(tex_row)
    
    tex_df = pd.DataFrame(latex_rows)
    latex_code = tex_df.to_latex(index=False, escape=False)
    
    with open(output_path, 'w') as f:
        f.write(latex_code)
        
    print(f"LaTeX table saved to {output_path}")