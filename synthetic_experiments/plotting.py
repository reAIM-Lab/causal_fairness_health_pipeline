import os
import numpy as np
import pickle as pk
from estimation import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import matplotlib.lines as mlines


def return_relative_errors(df):
    df['EYx0_Error%'] = rel_err(df['Estimated_EYx0'], df['EYx0'])
    df['EYx1_Error%'] = rel_err(df['Estimated_EYx1'], df['EYx1'])
    df['EYx0_sn_Error%'] = rel_err(df['Estimated_EYx0_sn'], df['EYx0'])
    df['EYx1_sn_Error%'] = rel_err(df['Estimated_EYx1_sn'], df['EYx1'])
    df['EYx1WVx0_Error%'] = rel_err(df['Estimated_EYx1WVx0'], df['EYx1Wx0'])
    df['EYx1WVx0_sn_Error%'] = rel_err(df['Estimated_EYx1WVx0_sn'], df['EYx1Wx0'])
    df['NDE_Error%'] = rel_err(df['Estimated_NDE'], df['NDE'])
    df['NIE_Error%'] = rel_err(df['Estimated_NIE'], df['NIE'])
    df['ExpSE_X0_Error%'] = rel_err(df['Estimated_ExpSE_x0'], df['expse_x0'])
    df['ExpSE_X1_Error%'] = rel_err(df['Estimated_ExpSE_x1'], df['expse_x1'])
    df['SE_Error%'] = rel_err(df['Estimated_SE'], df['SE'])
    df['NDE_sn_Error%'] = rel_err(df['Estimated_NDE_sn'], df['NDE'])
    df['NIE_sn_Error%'] = rel_err(df['Estimated_NIE_sn'], df['NIE'])
    df['ExpSE_X0_sn_Error%'] = rel_err(df['Estimated_ExpSE_x0_sn'], df['expse_x0'])
    df['ExpSE_X1_sn_Error%'] = rel_err(df['Estimated_ExpSE_x1_sn'], df['expse_x1'])
    df['SE_sn_Error%'] = rel_err(df['Estimated_SE_sn'], df['SE'])
    return df



def summarize_with_ci(df, group_cols, value_cols, ci=0.95):
    """
    Group by `group_cols` and compute mean + lower/upper CI for each col in `value_cols`.

    Returns a dataframe with columns like: col_mean, col_ci_low, col_ci_high
    """
    lower_q = (1 - ci) / 2
    upper_q = 1 - lower_q

    agg_funcs = {}
    for col in value_cols:
        agg_funcs[f"{col}_mean"] = pd.NamedAgg(column=col, aggfunc="mean")
        agg_funcs[f"{col}_ci_low"] = pd.NamedAgg(column=col, aggfunc=lambda x: x.quantile(lower_q))
        agg_funcs[f"{col}_ci_high"] = pd.NamedAgg(column=col, aggfunc=lambda x: x.quantile(upper_q))

    summary = df.groupby(group_cols).agg(**agg_funcs).reset_index()

    return summary


def get_best_interventions(df_raw):
    selected_segments = []
    df_baseline = df_raw[df_raw['intervention'] == 'No intervention'].copy()
    if not df_baseline.empty: selected_segments.append(df_baseline)

    tunable = [k for k in dict_interventions.keys() if k != 'No intervention']
    for intervention in tunable:
        subset = df_raw[df_raw['intervention'] == intervention]
        if subset.empty: continue
        avg_errors = subset.groupby('percent_dim_covered')['sum_error'].mean()
        if avg_errors.empty: continue
        best_pct = avg_errors.idxmin()
        best_subset = subset[subset['percent_dim_covered'] == best_pct].copy()
        best_subset['intervention'] = f"{intervention} ({int(best_pct * 100)}%)"
        selected_segments.append(best_subset)
        
    if not selected_segments: return pd.DataFrame()
    return pd.concat(selected_segments).sort_values(['n', 'sum_error'])

def get_pfi20_explicitly(df_raw):
    subset = df_raw[df_raw['intervention'] == 'Learn Y (PFI)']
    if subset.empty: return pd.DataFrame()
    pfi_20 = subset[np.isclose(subset['percent_dim_covered'], 0.2)].copy()
    if not pfi_20.empty:
        pfi_20['intervention'] = "Learn Y (PFI) (20%)"
        return pfi_20
    return pd.DataFrame()

def get_style_for_intervention(intervention_name):
    color = 'gray'
    marker = 'o'
    matched_key = None
    for key in base_colors.keys():
        if intervention_name.startswith(key):
            if matched_key is None or len(key) > len(matched_key):
                matched_key = key
    if matched_key:
        color = base_colors[matched_key]
        marker = base_markers.get(matched_key, 'o')
    return color, marker

def plot_in_axis(ax, df, metric_col, title, show_legend=False, legend_bbox=None):
    unique_interventions = df['intervention'].unique()
    linear_ticks = np.arange(0, 100001, 20000)
    
    for intervention in unique_interventions:
        color, marker = get_style_for_intervention(intervention)
        subset = df[df['intervention'] == intervention].sort_values('n')
        x = subset['n']
        y = subset[f'{metric_col}_mean']
        y_low = subset[f'{metric_col}_ci_low']
        y_high = subset[f'{metric_col}_ci_high']
        yerr = [y - y_low, y_high - y]
        
        ax.errorbar(
            x, y, yerr=yerr, marker=marker, color=color,
            label=intervention, linestyle='-',
            capsize=10, elinewidth=5, markeredgewidth=4
        )

    ax.set_xlabel('Number of Samples (n)', fontweight='bold')
    if metric_col == 'NDE_sn_Error%':
        ax.set_ylabel('NDE Error (%)', fontweight = 'bold')
    elif metric_col == 'NIE_sn_Error%':
        ax.set_ylabel('NIE Error (%)', fontweight = 'bold')
    ax.set_title(title, y=1, fontweight='bold', va='baseline')
    ax.set_xticks(linear_ticks)
    
    def tens_k_formatter(x, pos):
        if x == 0: return "0"
        return f'{int(x/1000)}k'
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(tens_k_formatter))
    
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if legend_bbox:
            # Place outside to the right
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=legend_bbox, frameon=True)
        else:
            # Default inside
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', frameon=True)

# --- PANEL DATA LOGIC ---
def get_no_intervention_dim_error():
    nointervention = 'results/estimates_largew/'
    nointervention_perf = []
    performance_cols = ['px_wz_auroc', 'px_wz_brier', 'y_wvzx0_auroc', 'y_wvzx0_brier', 'y_wvzx1_auroc', 'y_wvzx1_brier']
    
    for dimw in [500, 750, 1000, 1250, 1500, 1750]:
        dimz = 10 if dimw < 1000 else 30
        fpath = f'{nointervention}/estimated_effects_dimw{dimw}_dimz{dimz}.csv'
        if not os.path.exists(fpath): continue
            
        effects = pd.read_csv(fpath)
        effects['dimw'] = dimw
        mpath = f'{nointervention}/submodel_performances_dimw{dimw}_dimz{dimz}.csv'
        if os.path.exists(mpath):
            models = pd.read_csv(mpath)
            effects[performance_cols] = models[performance_cols]
        tpath = f'{true_effects_path}/true_effects_dimw{dimw}_dimz{dimz}.pkl'
        if os.path.exists(tpath):
            with open(tpath, 'rb') as f:
                dict_effects = pk.load(f)
            for k in dict_effects: effects[k] = dict_effects[k]
        nointervention_perf.append(effects)

    if not nointervention_perf: return pd.DataFrame()

    df = pd.concat(nointervention_perf)
    df = return_relative_errors(df)
    df[['NDE_sn_Error%', 'NIE_sn_Error%', 'SE_sn_Error%']] = np.abs(df[['NDE_sn_Error%', 'NIE_sn_Error%', 'SE_sn_Error%']])
    
    error_cols = ['NDE_sn_Error%', 'NIE_sn_Error%', 'SE_sn_Error%']
    grouped = df.groupby(['dimw', 'n'])[error_cols].agg(['mean', 'std', 'count']).reset_index()
    grouped.columns = ['dimw', 'n'] + [f'{col}_{stat}' for col in error_cols for stat in ['mean', 'std', 'count']]
    
    for col in error_cols:
        grouped[f'{col}_low'] = grouped[f'{col}_mean'] - 1.96 * grouped[f'{col}_std'] / np.sqrt(grouped[f'{col}_count'])
        grouped[f'{col}_high'] = grouped[f'{col}_mean'] + 1.96 * grouped[f'{col}_std'] / np.sqrt(grouped[f'{col}_count'])
    return grouped

def plot_dim_trend(ax, df, metric, title, show_legend=False):
    """Plots a specific metric vs Dim W (same style as Panel A)."""
    n_values = sorted(df['n'].unique())
    colors = plt.cm.PuBu(np.linspace(0.4, 1.0, len(n_values)))
    
    for i, n_val in enumerate(n_values):
        sub = df[df['n'] == n_val].sort_values('dimw')
        x = sub['dimw']
        y = sub[f'{metric}_mean']
        yerr = [y - sub[f'{metric}_low'], sub[f'{metric}_high'] - y]
        
        ax.errorbar(
            x, y, yerr=yerr,
            label=f"n={n_val}", color=colors[i], marker='o',
            linestyle='-', capsize=10, elinewidth=5, markeredgewidth=4
        )
    
    ax.set_title(title, y=1, fontweight='bold', va='baseline')
    ax.set_xlabel("Dimensionality of W", fontweight='bold')
    if metric == 'NDE_sn_Error%':
        ax.set_ylabel('NDE Error (%)', fontweight = 'bold')
    elif metric == 'NIE_sn_Error%':
        ax.set_ylabel('NIE Error (%)', fontweight = 'bold')
    elif metric == 'SE_sn_Error%':
        ax.set_ylabel('SE Error (%)', fontweight = 'bold')
    
    if show_legend:
        ax.legend(title="Num. Samples", loc='upper left', title_fontsize=40)