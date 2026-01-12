import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import os

# --- GLOBAL STYLE SETTINGS ---
def apply_publication_theme():
    """Sets global matplotlib parameters for publication-quality figures with LARGE text."""
    plt.rcParams.update({
        # Fonts
        'font.weight': 'bold',
        'font.size': 40,            # Base font size
        
        # Axes
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 40,       # LARGE Titles
        'axes.labelsize': 36,       # LARGE Axis Labels
        'axes.linewidth': 4,        # Thicker spines
        
        # Ticks
        'xtick.labelsize': 32,      # LARGE Tick Labels
        'ytick.labelsize': 32,
        'xtick.major.width': 4,     # Thicker ticks
        'ytick.major.width': 4,
        
        # Legend
        'legend.fontsize': 36,      # Readable Legend
        'legend.title_fontsize': 36,
        
        # Figure
        'figure.titlesize': 48,
        'figure.titleweight': 'bold',
        
        # Elements
        'lines.linewidth': 6,       # Thicker lines
        'lines.markersize': 20,     # Large markers
    })

# --- CONSTANTS ---
AVG_EFFECT_COLS = ['Estimated_NDE_sn', 'Estimated_NIE_sn', 'Estimated_SE_sn']
INDIVIDUAL_EFFECTS = ['Estimated_TE_sn', 'Estimated_NDE_sn', 'Estimated_NIE_sn', 'Estimated_SE_sn']

# --- DATA LOADING ---
def load_data(disease_name):
    """
    Loads selected_models.csv, separating the 'dataset' baseline row.
    """
    path = f"{disease_name}/selected_models.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")
        
    df = pd.read_csv(path)
    
    # Separate the 'dataset' baseline row
    dataset_row = df[df['experiment_name'] == 'Data'].iloc[0]
    models_df = df[df['experiment_name'] != 'Data'].copy()
    
    return models_df, dataset_row

# --- HELPERS ---
def assign_styles(df, order=None, manual_styles=None):
    """
    Assigns styles based on Experiment Prefix.
    - If 'order' is provided, experiments are processed in that sequence.
    - Same Prefix (before ':') -> Same Color.
    - Different Suffix -> Different Marker.
    """
    # Determine the processing order
    if order:
        # 1. Start with the user-provided order (filtering only those in the DF)
        experiments = [x for x in order if x in df['experiment_name'].unique()]
        # 2. Append any remaining experiments that weren't in the list (sorted alphabetically)
        seen = set(experiments)
        remaining = sorted([x for x in df['experiment_name'].unique() if x not in seen])
        experiments.extend(remaining)
    else:
        experiments = sorted(df['experiment_name'].unique())
    
    # 1. Group Experiments by Prefix (preserving order)
    groups = {}
    for exp in experiments:
        prefix = exp.split(":")[0].strip()
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(exp)
    
    # 2. Assign a unique color to each Group Prefix
    unique_prefixes = list(groups.keys())
    
    # Only sort prefixes alphabetically if NO order was provided.
    # If order WAS provided, we respect the discovery order of prefixes.
    if not order:
        unique_prefixes.sort()
    
    # Use a high-contrast palette
    palette = sns.color_palette("bright", n_colors=len(unique_prefixes)) 
    prefix_color_map = {prefix: palette[i] for i, prefix in enumerate(unique_prefixes)}
    
    # 3. Assign markers within the group
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
    
    style_map = {}
    
    for prefix, exp_list in groups.items():
        group_color = prefix_color_map[prefix]
        for i, exp_name in enumerate(exp_list):
            if manual_styles and exp_name in manual_styles:
                style_map[exp_name] = manual_styles[exp_name]
            else:
                style_map[exp_name] = {
                    'color': group_color,
                    'marker': markers[i % len(markers)]
                }
                
    return style_map
    
def calculate_avg_abs_effect(row, cols):
    """Calculates mean of absolute values for specified effect columns."""
    abs_sum = 0
    count = 0
    for col in cols:
        col_name = f"test_{col}_mean" 
        if col_name in row and not pd.isna(row[col_name]):
            abs_sum += abs(row[col_name])
            count += 1
    return abs_sum / count if count > 0 else np.nan

def add_subplot_label(ax, label):
    """Adds 'A.', 'B.' tags to the top-left of subplots."""
    # Adjusted font size for the subplot tag to match new theme
    trans = mtransforms.ScaledTranslation(-15/72, 7/72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, label, transform=ax.transAxes + trans,
            fontsize=40, fontweight='bold', va='bottom')