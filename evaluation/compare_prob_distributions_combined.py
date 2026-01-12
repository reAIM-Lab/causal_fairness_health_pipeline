import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import textwrap
import os

# --- 1. Global Configuration for Colors ---
MODEL_COLORS = {
    'Data': 'Black',
    'Baseline model': (0.00784313725490196, 0.24313725490196078, 1.0), # Blue
    'Causally fair resampling': (1.0, 0.48627450980392156, 0.0), # orange
    'Equalized odds': (0.6392156862745098, 0.6392156862745098, 0.6392156862745098), # grey
    'Greedy feature selection': (0.9098039215686274, 0.0, 0.043137254901960784),  # Red
    'Learning fair representations':  (0.9450980392156862, 0.2980392156862745, 0.7568627450980392), # pink
    'Demographic unawareness': (0.6235294117647059, 0.2823529411764706, 0.0),# brown
    '"Unbiased" feature selection': (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),  # purple
    'Path-specific inprocessing: All Effects': (0.10196078431372549, 0.788235294117647, 0.2196078431372549) # green 
}

def calculate_correlations(df, list_demos, list_models, alpha=0.05):
    """
    Calculates Pearson correlation and 95% CI between demographics and model outputs.
    """
    results = []
    for demo in list_demos:
        for model in list_models:
            temp_df = df[[demo, model]].dropna()
            
            if len(temp_df) < 3:
                results.append({
                    'demographic': demo,
                    'model': model,
                    'pearson_r': np.nan,
                    'ci_low': np.nan,
                    'ci_high': np.nan
                })
                continue
                
            r, _ = stats.pearsonr(temp_df[demo], temp_df[model])
            
            if np.abs(r) >= 1.0:
                ci_low, ci_high = r, r
            else:
                z = np.arctanh(r)
                se = 1 / np.sqrt(len(temp_df) - 3)
                z_crit = stats.norm.ppf(1 - alpha/2)
                
                ci_low = np.tanh(z - z_crit * se)
                ci_high = np.tanh(z + z_crit * se)
            
            results.append({
                'demographic': demo,
                'model': model,
                'pearson_r': r,
                'ci_low': ci_low,
                'ci_high': ci_high
            })
            
    return pd.DataFrame(results)

def process_disease_data(df, list_demos, list_models, output_prefix):
    """
    Calculates correlations and saves the CSV/LaTeX tables.
    Returns the correlation DataFrame.
    """
    corr_df = calculate_correlations(df, list_demos, list_models)
    
    os.makedirs(output_prefix, exist_ok=True)
    csv_filename = f"{output_prefix}/{output_prefix}_correlations.csv"
    corr_df.to_csv(csv_filename, index=False)
    print(f"[{output_prefix.upper()}] CSV saved to: {csv_filename}")
    
    def format_latex_entry(row):
        if pd.isna(row['pearson_r']): return "N/A"
        return f"{row['pearson_r']:.3g} ({row['ci_low']:.3g}, {row['ci_high']:.3g})"
    
    latex_df = corr_df.copy()
    latex_df['formatted'] = latex_df.apply(format_latex_entry, axis=1)
    latex_pivot = latex_df.pivot(index='demographic', columns='model', values='formatted')
    
    latex_filename = f"{output_prefix}/{output_prefix}_corrtable.tex"
    with open(latex_filename, 'w') as f:
        f.write(latex_pivot.to_latex(caption=f"Pearson Correlations ({output_prefix})", label=f"tab:corr_{output_prefix}"))
    print(f"[{output_prefix.upper()}] LaTeX table saved to: {latex_filename}")
    
    return corr_df

def plot_correlation_on_axis(ax, corr_df, list_demos, list_models, disease_title, plot_label=None):
    """
    Helper function to plot correlation bars on a given axis.
    """
    y_pos = np.arange(len(list_demos))
    wrapped_demos = ['\n'.join(textwrap.wrap(l, 18)) for l in list_demos]

    for j, model in enumerate(list_models):
        subset = corr_df[corr_df['model'] == model]
        subset = subset.set_index('demographic').reindex(list_demos).reset_index()
        
        offset = (j - (len(list_models) - 1) / 2) * 0.15
        y_vals = y_pos + offset
        
        x_err_lower = subset['pearson_r'] - subset['ci_low']
        x_err_upper = subset['ci_high'] - subset['pearson_r']
        color = MODEL_COLORS.get(model, 'black')
        
        ax.errorbar(
            x=subset['pearson_r'],
            y=y_vals,
            xerr=[x_err_lower, x_err_upper],
            fmt='o',
            label=model,
            capsize=10,
            elinewidth=5,
            markeredgewidth=4,
            markersize=20,
            color=color
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(wrapped_demos, fontweight='bold')
    ax.axvline(0, color='gray', linestyle='--', linewidth=4)
    ax.set_xlabel("Pearson Correlation (r)", fontweight='bold', labelpad=20)

    ax.set_title(f'{disease_title}: Sensitive attribute-\noutput correlation', fontweight='bold')
    
    if plot_label:
        ax.text(0, 1, f'({plot_label})', transform=ax.transAxes, 
                fontsize=40, fontweight='bold', va='bottom', ha='right')

    # Legend
    ax.legend(title="Interventions", 
              bbox_to_anchor=(0.5, -0.15), 
              loc='upper center', 
              borderaxespad=0., 
              frameon=False,
              ncol=1)

def create_combined_plot(plot_data_list, output_filename="combined_analysis_plot"):
    """
    Generates a 1x4 subplot figure.
    """
    plt.rcParams.update({
        'font.weight': 'bold', 'font.size': 30,
        'axes.titlesize': 40, 'axes.labelsize': 36,
        'xtick.labelsize': 30, 'ytick.labelsize': 30,
        'legend.fontsize': 24, 'legend.title_fontsize': 28,
        'lines.linewidth': 6, 'lines.markersize': 20,
        'figure.titlesize': 48
    })

    fig, axes = plt.subplots(1, 5, figsize=(60, 18), 
                             gridspec_kw={'wspace': 0.3, 'bottom': 0.2, 
                                          'width_ratios': [1, 1, 0.15, 1, 1]})
    
    # Mapping for 4 datasets to 5 columns (skipping index 2)
    col_indices = [0, 1, 3, 4]

    # Hide the spacer axis
    axes[2].axis('off')

    if len(plot_data_list) != 4:
        raise ValueError("Expected exactly 4 datasets for this layout.")    

    for i, data in enumerate(plot_data_list):
        target_col = col_indices[i]
        plot_correlation_on_axis(
            axes[target_col], 
            data['corr_df'], 
            data['demos'], 
            data['models'], 
            data['disease_name'], 
            data['label']
        )

    plt.savefig(f"{output_filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_filename}.pdf", dpi=300, bbox_inches='tight')
    print(f"Combined plots saved to: {output_filename}.png and .pdf")

def save_individual_plot(data):
    """
    Generates and saves a single plot for a disease with overrides.
    """
    plt.rcParams.update({
        'font.weight': 'bold', 'font.size': 30,
        'axes.titlesize': 40, 'axes.labelsize': 36,
        'xtick.labelsize': 30, 'ytick.labelsize': 30,
        'legend.fontsize': 24, 'legend.title_fontsize': 28,
        'lines.linewidth': 6, 'lines.markersize': 20,
        'figure.titlesize': 48
    })
    
    fig_height = len(data['demos']) * 2 + 8
    fig, ax = plt.subplots(figsize=(16, fig_height))
    
    # 1. Base Plot
    plot_correlation_on_axis(
        ax, 
        data['corr_df'], 
        data['demos'], 
        data['models'], 
        data['disease_name'], 
        plot_label=None
    )
    
    # 2. Override Title
    ax.set_title("Correlation: sensitive attribute \nvs. data/model outputs", 
                 pad=30, fontweight='bold', fontsize=40)
    
    # 3. Override Legend (Move to Side)
    # Get existing handles/labels to ensure we don't lose them
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels,
              title="Interventions",
              bbox_to_anchor=(1.05, 1), 
              loc='upper left', 
              borderaxespad=0., 
              frameon=False)
    
    # 4. Rotate X-Ticks 45 degrees
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    folder = data['disease_name'].lower()
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/{folder}_correlation_plot"
    
    # Using bbox_inches='tight' will accommodate the external legend automatically
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename}.pdf", dpi=300, bbox_inches='tight')
    print(f"[{data['disease_name']}] Individual plot saved to {filename}")
    plt.close()

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
collected_plot_data = []

# --- 1. AMI ---
disease = 'ami'
print(f"\n--- Processing {disease} ---")
"""
DATA PATH
"""
try:
    df_data = pd.read_csv(f'{path}/{dataset_prefix}features.csv', index_col = 0)
    df_outputs = df_data[['PID_unique', 'person_id', 'prediction_time', 'boolean_value', 'is_Male', 'is_Female']]
    df_outputs.rename({'is_Male':'Male', 'is_Female':'Female','boolean_value': 'Data'}, axis=1, inplace=True)

    list_demos = ['Male', 'Female']
    list_models = ['Data', 'Baseline model', 'Greedy feature selection', 'Causally fair resampling', '"Unbiased" feature selection']
    model_folders = 'SPECIFY MODEL FOLDERS'
    for model_name, model_folder in model_folders:
        try:
            df_test = pd.read_csv(f'{model_folder}/test_outputs.csv')
            df_test.rename({'y_pred':model_name}, axis=1, inplace=True)
            if 'pred_time' in df_outputs:
                df_outputs.drop('pred_time', axis=1, inplace=True)
            df_outputs = df_outputs.merge(df_test[['person_id', 'pred_time', model_name]], how = 'inner', left_on = ['person_id', 'prediction_time'], right_on = ['person_id', 'pred_time'])
        except FileNotFoundError:
            print(f"Skipping model {model_name} (file not found)")

    corr_ami = process_disease_data(df_outputs, list_demos, list_models, output_prefix=disease)
    collected_plot_data.append({
        'disease_name': 'AMI', 
        'corr_df': corr_ami,
        'demos': list_demos,
        'models': [m for m in list_models if m in df_outputs.columns],
        'label': 'A'
    })
except FileNotFoundError:
    print(f"Skipping {disease} (base data not found)")


# --- 2. SLE ---
disease = 'sle'
print(f"\n--- Processing {disease} ---")
# PATHS 
try:
    df_data = pd.read_csv(f'{path}/{dataset_prefix}features.csv', index_col = 0)
    df_outputs = df_data[['PID_unique', 'person_id', 'prediction_time', 'boolean_value', 'is_Male', 'is_Female']]
    df_outputs.rename({'is_Male':'Male', 'is_Female':'Female','boolean_value': 'Data'}, axis=1, inplace=True)

    list_demos = ['Male', 'Female']
    list_models = ['Data', 'Remove demographics', '"Unbiased" feature selection', 'Causally fair resampling']
    model_folders = 'SPECIFY MODEL FOLDERS'
    for model_name, model_folder in model_folders:
        try:
            df_test = pd.read_csv(f'{model_folder}/test_outputs.csv')
            df_test.rename({'y_pred':model_name}, axis=1, inplace=True)
            df_outputs = df_outputs.merge(df_test[['person_id', 'pred_time', model_name]], how = 'inner', left_on = ['person_id', 'prediction_time'], right_on = ['person_id', 'pred_time'])
        except FileNotFoundError:
            print(f"Skipping model {model_name}")

    corr_sle = process_disease_data(df_outputs, list_demos, list_models, output_prefix=disease)
    collected_plot_data.append({
        'disease_name': 'SLE',
        'corr_df': corr_sle,
        'demos': list_demos,
        'models': [m for m in list_models if m in df_outputs.columns],
        'label': 'B'
    })
except FileNotFoundError:
    print(f"Skipping {disease}")


# --- 3. T2DM ---
disease = 't2dm'
print(f"\n--- Processing {disease} ---")
# PATHS
try:
    df_data = pd.read_csv(f'{path}/{dataset_prefix}features.csv', index_col = 0)
    df_outputs = df_data[['PID_unique', 'person_id', 'prediction_time', 'boolean_value', 'is_Black', 'is_White', 'is_MissingRace', 'race_concept_id']]
    df_outputs.rename({'is_Black':'Black', 'is_White':'White', 'is_MissingRace':'Missing/Other Race', 'is_OtherRace':'Other Race', 'is_Asian': 'Asian', 'boolean_value': 'Data'}, axis=1, inplace=True)

    # add other races
    df_outputs[['Asian', 'Native Hawaiian/Pacific Islander', 'American Indian/Alaskan Native']] = 0
    df_outputs.loc[df_outputs['race_concept_id'] == 8515, 'Asian'] = 1
    df_outputs.loc[df_outputs['race_concept_id'] == 8557, 'Native Hawaiian/Pacific Islander'] = 1
    df_outputs.loc[df_outputs['race_concept_id'] == 8552, 'Missing/Other Race'] = 1
    df_outputs.loc[df_outputs['race_concept_id'] == 8522, 'Missing/Other Race'] = 1
    df_outputs.loc[df_outputs['race_concept_id'] == 8657, 'American Indian/Alaskan Native'] = 1

    list_demos = ['Missing/Other Race', 'Black', 'White', 'Asian', 'Native Hawaiian/Pacific Islander', 'American Indian/Alaskan Native']
    list_models = ['Data', 'Causally fair resampling', 'Path-specific inprocessing: All Effects']
    model_folders = 'SPECIFY MODEL FOLDERS'

    for model_name, model_folder in model_folders:
        try:
            df_test = pd.read_csv(f'{model_folder}/test_outputs.csv')
            df_test.rename({'y_pred':model_name}, axis=1, inplace=True)
            df_outputs = df_outputs.merge(df_test[['person_id', 'pred_time', model_name]], how = 'inner', left_on = ['person_id', 'prediction_time'], right_on = ['person_id', 'pred_time'])
        except FileNotFoundError:
            print(f"Skipping model {model_name}")

    corr_t2dm = process_disease_data(df_outputs, list_demos, list_models, output_prefix=disease)
    collected_plot_data.append({
        'disease_name': 'T2DM',
        'corr_df': corr_t2dm,
        'demos': list_demos,
        'models': [m for m in list_models if m in df_outputs.columns],
        'label': 'C'
    })
except FileNotFoundError:
    print(f"Skipping {disease}")


# --- 4. SCZ ---
disease = 'scz'
print(f"\n--- Processing {disease} ---")
# PATHS
try:
    df_pop = pd.read_csv(f'{data_path}/population_2dx.csv')
    df_pop[['Missing Race', 'Black', 'White']] = 0
    df_pop.loc[df_pop['race_concept_id']==0, 'Missing Race'] = 1
    df_pop.loc[df_pop['race_concept_id']==8516, 'Black'] = 1
    df_pop.loc[df_pop['race_concept_id']==8527, 'White'] = 1
    df_pop.rename({'sz_flag':'Data'}, axis=1, inplace=True)

    df_outputs = df_pop[['person_id', 'Missing Race', 'Black', 'White', 'Data']]
    model_folders = 'SPECIFY MODEL FOLDERS'
    
    for model_name, model_folder in model_folders:
        try:
            df_test = pd.read_csv(f'{model_folder}/test_outputs.csv')
            df_test.rename({'y_pred':model_name}, axis=1, inplace=True)
            df_outputs = df_outputs.merge(df_test[['person_id', model_name]], how = 'inner', on = 'person_id')
        except FileNotFoundError:
            print(f"Skipping model {model_name}")

    # Explicitly use newline for wrapping logic
    df_outputs.rename(columns={'Missing Race': 'Missing\nRace'}, inplace=True)
    list_demos = ['Missing\nRace', 'Black', 'White']
    list_models = ['Data', 'Baseline model', 'Greedy feature selection', '"Unbiased" feature selection']

    corr_scz = process_disease_data(df_outputs, list_demos, list_models, output_prefix=disease)
    collected_plot_data.append({
        'disease_name': 'SCZ',
        'corr_df': corr_scz,
        'demos': list_demos,
        'models': [m for m in list_models if m in df_outputs.columns],
        'label': 'D'
    })
except FileNotFoundError:
    print(f"Skipping {disease}")


# --- Generate Final Plots ---
if len(collected_plot_data) > 0:
    print("\nGenerating combined plot...")
    create_combined_plot(collected_plot_data, output_filename="all_diseases_correlations")
    
    print("\nGenerating individual plots...")
    for data in collected_plot_data:
        save_individual_plot(data)
else:
    print("\nNo data collected. Exiting.")