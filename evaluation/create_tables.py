import pandas as pd
from metrics import get_performance_row
from processing import summarize_causal_effects, select_best_model, generate_latex_table
import os

# --- CONFIG ---
# List of causal effects you want to process
CAUSAL_COLS = ['Estimated_TE_sn', 'Estimated_NDE_sn', 'Estimated_NIE_sn', 'Estimated_SE_sn', 'Estimated_ExpSE_x1_sn', 'Estimated_ExpSE_x0_sn']

def main(disease):
    # 1. Load the master config
    # Columns: experiment_name, model_path, effect_path, featureset
    input_csv = f"{disease.lower()}/{disease.lower()}_experiments_config.csv" 
    if not os.path.exists(input_csv):
        print(f"Please create {input_csv} first.")
        return
        
    config_df = pd.read_csv(input_csv)
    all_results = []

    print("Starting pipeline processing...")

    # 2. Loop through config to build the Massive CSV
    for idx, row in config_df.iterrows():
        print(f"Processing {row['experiment_name']}...")
        
        # A. Performance Metrics
        val_perf = get_performance_row(row['model_path'], 'val')
        test_perf = get_performance_row(row['model_path'], 'test')
        
        # B. Causal Effects
        try:
            # We need to load all three splits and combine them
            splits = ['train', 'test', 'val'] # check if your files use 'val' or 'validation'
            dfs = []

            for split in splits:
                # Construct filename: e.g. estimated_effects_splittrain_features123.csv
                effect_file = f"{row['effect_path']}/estimated_effects_split{split}_features{row['featureset']}.csv"
                
                if os.path.exists(effect_file):
                    temp_df = pd.read_csv(effect_file)
                    # IMPORTANT: Ensure the dataframe has a 'split' column so summarize_causal_effects knows which is which
                    # If your CSVs don't have a 'split' column, we force it here:
                    temp_df['split'] = split if split != 'val' else 'validation' # standardizing 'val' to 'validation' for processing.py logic
                    dfs.append(temp_df)
                else:
                    print(f"  Warning: Missing effect file {effect_file}")

            if dfs:
                # Concatenate all splits into one big dataframe
                full_effects_df = pd.concat(dfs, ignore_index=True)
                effects_data = summarize_causal_effects(full_effects_df, CAUSAL_COLS)
            else:
                print(f"  Error: No effect files found for {row['experiment_name']}")
                effects_data = {}

        except Exception as e:
            print(f"  Error processing effects for {row['experiment_name']}: {e}")
            effects_data = {}

        # C. Combine
        combined_row = row.to_dict()
        combined_row.update(val_perf)
        combined_row.update(test_perf)
        combined_row.update(effects_data)
        
        all_results.append(combined_row)

    # 3. Save Massive CSV
    full_df = pd.DataFrame(all_results)
    full_df.to_csv(f"{disease.lower()}/full_results_dump.csv", index=False)
    print("Saved artifacts/full_results_dump.csv")

    # 4. Model Selection
    # Group by experiment_name, select best based on Val BCE and Val Causal Bias
    selected_df = select_best_model(full_df)
    selected_df.to_csv(f"{disease.lower()}/selected_models.csv", index=False)
    print("Saved artifacts/selected_models.csv")

    # 5. Generate LaTeX Table
    generate_latex_table(selected_df, f"{disease.lower()}/publication_table.tex")
    print("Pipeline complete.")

if __name__ == "__main__":
    # Ensure artifacts dir exists
    main("ami")
    main("sle")
    main("t2dm")
    main("scz")
    
