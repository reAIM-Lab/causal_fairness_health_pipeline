## Task-Specific Modeling

This folder contains training scripts for the baseline model and fairness interventions
- Baseline model, demographic unawareness: **`run_logreg_baseline_models`**  
  Specify 'du' or 'da' unawareness for the demographic unaware model (du) and the baseline model (da). 
- Path-specific inprocessing: **`run_pathspecific_inprocessing.py`**. This is an implementation of: https://github.com/dplecko/CFA/tree/main
- "Unbiased" feature selection: run **`logreg_bias_importance.py`** to generate SMD for each feature, and then run **`run_unbiased_features_logreg.py`** to train models. 
- "Unbiased" feature selection: run **`logreg_bias_importance.py`** to generate importance and SMD for each feature, and then run **`run_greedy_features_logreg.py`** to train models. 
- Equalized odds: **`run_eqodds_logreg.py`**  
- LFR: **`lfr/run_lfr.py`**: This is a GPU-based implementation of: https://github.com/Trusted-AI/AIF360/blob/main/aif360/algorithms/preprocessing/lfr.py

## Feature Importance
- Run **`run_AFO_Explainer.py`** followed by **`get_complete_importance.py`** to calculate AFO-based feature importance in this temporal model. Code is adapted from: https://github.com/sanatonek/time_series_explainability. The feature importance here is used for causal effect estimation and for greedy feature selection. 