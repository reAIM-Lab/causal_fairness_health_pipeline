import os
import numpy as np
import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import re
import string

# Assuming these imports exist in your environment
from estimation import * 
from plotting import * 

# --- 1. Global Style Settings (UPDATED FOR VISIBILITY) ---
# Increased sizes for legibility at 75% zoom
font = {'weight': 'bold', 'size': 24}  # Increased from 18
matplotlib.rc('font', **font)
plt.rcParams.update({
    'axes.titlesize': 40,   
    'axes.labelsize': 36,   
    'xtick.labelsize': 30,  
    'ytick.labelsize': 30,  
    'legend.fontsize': 36,  
    'lines.linewidth': 5,   
    'lines.markersize': 20, 
    'figure.titlesize': 40  
})

true_effects_path = 'PATH TO DATASET'

# --- Configuration ---
dict_interventions = {
    'No intervention': 'results/estimates_largew', 
    'Learn W (autoencoder)': 'autoencoder_features/results/estimates_largew',
    'Learn Y (PFI)': 'important_features_pfi/results_largew',
    'Learn Y (Lasso)': 'important_features_lasso/results_largew',
    'Learn X (mean difference)': 'most_unfair_features/results/estimates_largew'
}

base_colors = {
    'No intervention': 'black',
    'Learn W (autoencoder)': 'green',
    'Learn Y (PFI)': 'red',
    'Learn Y (Lasso)': 'purple',
    'Learn X (mean difference)': 'orange',
}

base_markers = {
    'No intervention': 'o',                
    'Learn W (autoencoder)': 's',          
    'Learn Y (PFI)': 'D',                  
    'Learn Y (Lasso)': 'D',                
    'Learn X (mean difference)': 'P',      
}

list_dimws = [500, 750, 1000, 1250, 1500, 1750]
candidate_percentages = [0.2, 0.4, 0.5, 0.6, 0.8]
n_values_for_lineplot = [10000, 50000, 75000, 100000]

def load_dimw_files(path_dict, dimw):
    if dimw < 1000: dimz = 10
    else: dimz = 30

    estimated_effects_pattern = f"estimated_effects_dimw{dimw}"
    submodel_performance_pattern = f"submodel_performances_dimw{dimw}"
    reduced_dim_pattern = re.compile(rf"(\d+)_dimz{dimz}.csv")

    list_estimated_effects = []
    list_submodel_effects = []
    
    for key, folder in path_dict.items():
        if not os.path.exists(folder): continue
        for fname in os.listdir(folder):
            if fname.startswith(estimated_effects_pattern) and fname.endswith(f'dimz{dimz}.csv'):
                try:
                    effects = pd.read_csv(os.path.join(folder, fname))
                    effects['dimw'] = dimw
                    effects['dimz'] = dimz
                    
                    true_eff_file = f'{true_effects_path}/true_effects_dimw{dimw}_dimz{dimz}.pkl'
                    if os.path.exists(true_eff_file):
                        with open(true_eff_file, 'rb') as f:
                            dict_effects = pk.load(f)
                        for k in dict_effects:
                            effects[k] = dict_effects[k]
                    
                    match = reduced_dim_pattern.search(fname)
                    effects['reduced_dim'] = int(match.group(1)) if match else None
                    effects['intervention'] = key
                    list_estimated_effects.append(effects)
                except Exception as e: print(e)

            if fname.startswith(submodel_performance_pattern):
                try:
                    submodels = pd.read_csv(os.path.join(folder, fname))
                    match = reduced_dim_pattern.search(fname)
                    submodels['reduced_dim'] = int(match.group(1)) if match else None
                    submodels['intervention'] = key
                    submodels['dimw'] = dimw
                    list_submodel_effects.append(submodels)
                except Exception as e: print(e)
    
    if not list_estimated_effects: return pd.DataFrame(), pd.DataFrame()

    df_est = pd.concat(list_estimated_effects)
    df_sub = pd.concat(list_submodel_effects)

    df_est = return_relative_errors(df_est)
    df_est[['NDE_sn_Error%', 'NIE_sn_Error%', 'SE_sn_Error%']] = np.abs(df_est[['NDE_sn_Error%', 'NIE_sn_Error%', 'SE_sn_Error%']])
    
    df_est = summarize_with_ci(df_est, ['intervention', 'reduced_dim', 'n'], ['NDE_sn_Error%', 'NIE_sn_Error%', 'SE_sn_Error%'])
    df_sub = summarize_with_ci(df_sub, ['intervention', 'reduced_dim', 'n'], ['y_wvzx0_auroc', 'y_wvzx1_auroc', 'px_wz_auroc', 'y_wvzx0_brier', 'y_wvzx1_brier', 'px_wz_brier'])
    
    return df_est, df_sub

# --- MAIN EXECUTION ---
data_store = {}
out_dir = 'figures/'
os.makedirs(out_dir, exist_ok=True)

for dimw in list_dimws:
    print(f"Processing dimw={dimw}...")
    df_est, df_sub = load_dimw_files(dict_interventions, dimw)
    if df_est.empty: continue
    df_raw = df_est.merge(df_sub, how='outer', on=['intervention', 'reduced_dim', 'n'])
    df_raw['dimw'] = dimw
    df_raw['percent_dim_covered'] = df_raw['reduced_dim'] / df_raw['dimw']
    df_raw['sum_error'] = (df_raw['NDE_sn_Error%_mean'] + df_raw['NIE_sn_Error%_mean'] + df_raw['SE_sn_Error%_mean'])
    df_raw = df_raw[(df_raw['percent_dim_covered'].isin(candidate_percentages)) | (df_raw['intervention'] == 'No intervention')]
    data_store[dimw] = get_best_interventions(df_raw)
    df_pfi20 = get_pfi20_explicitly(df_raw)
    df_best = data_store[dimw]
    if not df_pfi20.empty and not df_best.empty:
        df_csv = pd.concat([df_best, df_pfi20]).drop_duplicates(subset=['intervention', 'n']).sort_values(['n', 'sum_error'])
    else: df_csv = df_best
    df_csv.to_csv(f'{out_dir}/tabular_estimates_best_plus_pfi20_dimw{dimw}.csv')

# --- 2. Generate 4x2 Plot ---
rows_dims = [750, 1000, 1250, 1500]
fig1, axes1 = plt.subplots(4, 2, figsize=(32, 36))
letters = list(string.ascii_uppercase)
letter_idx = 0

for i, d in enumerate(rows_dims):
    if d not in data_store: continue
    df = data_store[d]
    
    ax_nde = axes1[i, 0]
    plot_in_axis(ax_nde, df, 'NDE_sn_Error%', f'Dim. Reduction \nfrom |W| = {d} (NDE)')
    ax_nde.text(-0.1, 1.05, f'({letters[letter_idx]})', transform=ax_nde.transAxes, size=40, weight='bold', va='baseline')
    letter_idx += 1
    
    ax_nie = axes1[i, 1]
    plot_in_axis(ax_nie, df, 'NIE_sn_Error%', f'Dim. Reduction \nfrom |W| = {d} (NIE)', show_legend=True, legend_bbox=(1.05, 1))
    ax_nie.text(-0.1, 1.05, f'({letters[letter_idx]})', transform=ax_nie.transAxes, size=40, weight='bold', va='baseline')
    letter_idx += 1

plt.tight_layout()
plt.savefig(f'{out_dir}/figure_4x2_dims_750_1500.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Saved 4x2 plot.")

# --- 3. Generate 1x2 Plot (W=1750) ---
dim_single = 1750
if dim_single in data_store:
    fig2, axes2 = plt.subplots(1, 2, figsize=(36, 8)) 
    letters_single = ['(A)', '(B)']
    df = data_store[dim_single]
    
    plot_in_axis(axes2[0], df, 'NDE_sn_Error%', f'Dim. Reduction \nfrom |W| = {dim_single} (NDE)')
    axes2[0].text(0, 1.01, letters_single[0], transform=axes2[0].transAxes, size=40, weight='bold', va='baseline')

    # UPDATED: Place legend on the side
    plot_in_axis(axes2[1], df, 'NIE_sn_Error%', f'Dim. Reduction \nfrom |W| = {dim_single} (NIE)', 
                 show_legend=True, legend_bbox=(1.05, 1))
    axes2[1].text(0, 1.01, letters_single[1], transform=axes2[1].transAxes, size=40, weight='bold', va='baseline')

    plt.tight_layout()
    plt.savefig(f'{out_dir}/figure_1x2_dim_1750.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 1x2 plot.")


# --- 4. Generate NEW 1x3 Plot (No Intervention Error vs Dim) ---
# Panel A: NDE, Panel B: NIE, Panel C: SE
df_no_int = get_no_intervention_dim_error()
if not df_no_int.empty:
    fig3, axes3 = plt.subplots(1, 3, figsize=(40, 10)) 
    letters_tri = ['(A)', '(B)', '(C)']
    
    # Panel A: NDE
    plot_dim_trend(axes3[0], df_no_int, 'NDE_sn_Error%', 'No Intervention: NDE Error')
    axes3[0].text(0, 1.01, letters_tri[0], transform=axes3[0].transAxes, size=40, weight='bold', va='baseline')
    
    # Panel B: NIE
    plot_dim_trend(axes3[1], df_no_int, 'NIE_sn_Error%', 'No Intervention: NIE Error')
    axes3[1].text(0, 1.01, letters_tri[1], transform=axes3[1].transAxes, size=40, weight='bold', va='baseline')
    
    # Panel C: SE
    plot_dim_trend(axes3[2], df_no_int, 'SE_sn_Error%', 'No Intervention: SE Error', show_legend=True)
    axes3[2].text(0, 1.01, letters_tri[2], transform=axes3[2].transAxes, size=40, weight='bold', va='baseline')

    plt.tight_layout()
    plt.savefig(f'{out_dir}/figure_1x3_no_intervention_errors.pdf', dpi=1200, bbox_inches='tight')
    plt.close()
    print("Saved 1x3 No Intervention plot.")

