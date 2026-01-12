## Task-Specific Modeling

This folder contains training scripts for the baseline model and fairness interventions
- Baseline model, "unbiased" feature selection, greedy feature selection: **`run_transformer_training.py`**  
  Specify which data loader should be used (corresponding to a given feature selection strategy/full dataset for the baseline model) and run to train an encoder-only transformer model
- Demographic-Unaware Model: **`demo_unaware/run_du_transformer_training.py`**. 
- Path-specific inprocessing: **`pathspecific_inprocessing.py`**. This is an implementation of: https://github.com/dplecko/CFA/tree/main
- Equalized odds: **`run_transformer_training_eqodds.py`**  
- LFR: **`lfr/run_lfr.py`**: This is a GPU-based implementation of: https://github.com/Trusted-AI/AIF360/blob/main/aif360/algorithms/preprocessing/lfr.py

## Feature Importance
- Run **`run_AFO_Explainer.py`** followed by **`get_complete_importance.py`** to calculate AFO-based feature importance in this temporal model. Code is adapted from: https://github.com/sanatonek/time_series_explainability. The feature importance here is used for causal effect estimation and for greedy feature selection. 