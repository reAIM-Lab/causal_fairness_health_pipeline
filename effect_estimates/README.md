## Effect Estimation

We robustly estimate path-specific causal effects, based off of code from: https://github.com/reAIM-Lab/PSE-Pulse-Oximetry
The environment file for path-specific effect estimation is included in this folder. 
- **`run_raw_data.py`**  
  Generates causal effect estimations for data across all tasks. 
- **`calc_pfi.py`**  
  Identifies the most important features for outcome model prediction using permutation feature analysis (should be used for the foundation model tasks). 
- **`run_models_foundation_model.py`**  
  Generates causal effect estimations for foundation model outputs (AMI, SLE, T2DM). Add the paths to the model outputs (baseline model and/or fairness interventions) and run for sensitive attribute = race or gender
- **`run_model_effects_taskspecific.py`**  
  Generates causal effect estimations for task-specific (SCZ) model outputs. Add the paths to the model outputs (baseline model and/or fairness interventions) and run for sensitive attribute = race. 
- **`run_pxz_scores.py`**  
  Generates the propensity score P(X|Z) for use in the Plecko et al. path-specific inprocessing fairness intervention. 
