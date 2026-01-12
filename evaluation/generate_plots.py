import matplotlib.pyplot as plt
import string
import sys
import os
import viz_utils as utils  # Importing the file created above

def get_plot_order(df, order=None):
    """Helper to determine the final list of experiments to plot."""
    if order:
        # Use provided order, filtering for existence
        exp_list = [x for x in order if x in df['experiment_name'].unique()]
        # Append anything missing from the order list
        seen = set(exp_list)
        rest = sorted([x for x in df['experiment_name'].unique() if x not in seen])
        exp_list.extend(rest)
        return exp_list
    else:
        return sorted(df['experiment_name'].unique())

def plot_1_avg_tradeoff(df, dataset_row, style_map, output_dir, order=None):
    """
    Plot 1: AUROC vs Average Absolute Causal Effect
    """
    fig, ax = plt.subplots(figsize=(24, 16))
    
    # Calculate X-axis
    df['avg_abs_effect'] = df.apply(lambda row: utils.calculate_avg_abs_effect(row, utils.AVG_EFFECT_COLS), axis=1)
    
    # Calculate Baseline X-axis
    ds_val = 0
    count = 0
    for col in utils.AVG_EFFECT_COLS:
        val = dataset_row.get(f'test_{col}_mean', dataset_row.get(f'mean_{col}_mean'))
        if val is not None:
            ds_val += abs(val)
            count += 1
    ds_avg_effect = ds_val / count if count > 0 else 0

    # Reference Lines
    ax.axvline(0, color='black', linestyle='-', linewidth=5, alpha=0.8) 
    ax.axvline(ds_avg_effect, color='red', linestyle='--', linewidth=5, label='Dataset Bias')

    # Determine Order
    exp_list = get_plot_order(df, order)
    groups = df.groupby('experiment_name')

    # Plot Models in specific order
    for exp_name in exp_list:
        if exp_name not in groups.groups: continue
        group = groups.get_group(exp_name)
        
        style = style_map[exp_name]
        row = group.iloc[0]
        
        y = row['test_auroc_mean']
        y_err = [[y - row['test_auroc_ci_low']], [row['test_auroc_ci_high'] - y]]
        
        ax.errorbar(
            x=row['avg_abs_effect'], y=y, yerr=y_err,
            fmt=style['marker'], color=style['color'], label=exp_name,
            capsize=10, capthick=4, elinewidth=4, markersize=20
        )

    ax.set_xlabel("Average Absolute Causal Effect")
    ax.set_ylabel("AUROC")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=32)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/plot1_avg_tradeoff.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_dir}/plot1_avg_tradeoff.pdf", bbox_inches='tight')
    print(f"Saved Plot 1 to {output_dir}")
    plt.close()

def plot_2_individual_tradeoffs(df, dataset_row, style_map, output_dir, order=None):
    """
    Plot 2: 4 Subplots (1x4) for TE, NDE, NIE, SE vs AUROC
    """
    #fig, axes = plt.subplots(1, 4, figsize=(48, 12), sharey=True)
    fig, axes = plt.subplots(1, 4, figsize=(48, 10), sharey=True, gridspec_kw={'wspace': 0.05})
    labels = list(string.ascii_uppercase)
    
    # Determine Order
    exp_list = get_plot_order(df, order)
    groups = df.groupby('experiment_name')
    
    for idx, effect_col in enumerate(utils.INDIVIDUAL_EFFECTS):
        ax = axes[idx]
        utils.add_subplot_label(ax, f"({labels[idx]})")
        
        # Dataset Reference
        ds_val = dataset_row.get(f'test_{effect_col}_mean', dataset_row.get(f'mean_{effect_col}_mean'))
        ax.axvline(0, color='black', linestyle='-', linewidth=5, alpha=0.8)
        if ds_val is not None:
            ax.axvline(ds_val, color='red', linestyle='--', linewidth=5)
            
        # Plot Models in specific order
        for exp_name in exp_list:
            if exp_name not in groups.groups: continue
            group = groups.get_group(exp_name)
            
            style = style_map[exp_name]
            row = group.iloc[0]
            
            y = row['test_auroc_mean']
            y_err = [[y - row['test_auroc_ci_low']], [row['test_auroc_ci_high'] - y]]
            
            x = row[f'test_{effect_col}_mean']
            x_low = row[f'min_{effect_col}_ci_low']
            x_high = row[f'max_{effect_col}_ci_high']
            x_err = [[x - x_low], [x_high - x]]
            
            ax.errorbar(
                x=x, y=y, xerr=x_err, yerr=y_err,
                fmt=style['marker'], color=style['color'],
                capsize=10, capthick=4, elinewidth=4, markersize=20
            )
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
        clean_name = effect_col.replace('Estimated_', '').replace('_sn', '')
        long_names = {
            'TE': 'Total Effect', 
            'NDE': 'Natural Direct Effect', 
            'NIE': 'Natural Indirect Effect', 
            'SE': 'Spurious Effect'
        }
        ax.set_xlabel(long_names.get(clean_name, clean_name))
        ax.set_title(f'Performance tradeoff: \n{long_names.get(clean_name, clean_name)}')
        
        if idx == 0:
            ax.set_ylabel("AUROC")
        
    # Manual Legend Construction (Using exp_list for order)
    legend_elements = []
    for exp_name in exp_list:
        style = style_map[exp_name]
        line = plt.Line2D([0], [0], marker=style['marker'], color=style['color'], label=exp_name, 
                          markersize=24, linestyle='None')
        legend_elements.append(line)
    
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=5, label='Dataset Bias'))
    # fig.legend(handles=legend_elements, bbox_to_anchor=(1.01, 0.5), loc='center left', borderaxespad=0., fontsize=32)
    axes[-1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0., fontsize=32)
    
    # plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/plot2_individual_tradeoffs.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_dir}/plot2_individual_tradeoffs.pdf", bbox_inches='tight')
    print(f"Saved Plot 2 to {output_dir}")
    plt.close()

def main(disease_string):
    utils.apply_publication_theme()
    print(f"Generating figures for: {disease_string}")
    output_dir = disease_string

    # --- DEFINE YOUR CUSTOM ORDER HERE ---
    # Example order. Replace these strings with your actual experiment names.
    # Any names found in the CSV but NOT in this list will be appended alphabetically at the end.
    custom_order = ['Dataset bias', 'Baseline model', 'Causally fair resampling',
                    'Path-specific inprocessing: NDE', 'Path-specific inprocessing: NIE', 'Path-specific inprocessing: SE', 'Path-specific inprocessing: All Effects',
                    'Greedy feature selection', '"Unbiased" feature selection', 'Demographic unawareness', 
                    'Learning fair representations', 'Equalized Odds']

    try:
        models_df, dataset_row = utils.load_data(disease_string)
        
        # Pass order to style assignment to ensure colors follow the grouping order
        style_map = utils.assign_styles(models_df, order=custom_order)
        
        plot_1_avg_tradeoff(models_df, dataset_row, style_map, output_dir, order=custom_order)
        plot_2_individual_tradeoffs(models_df, dataset_row, style_map, output_dir, order=custom_order)
        
        print("Visualization pipeline complete.")
    except Exception as e:
        print(f"Error for {disease_string}: {e}")


if __name__ == "__main__":
    main("ami")
    main("sle")
    main("t2dm")
    main("scz")



