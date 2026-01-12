## Preprocessing the data

This folder contains scripts for processing the data for each of the models

### Task-Specific Model: Schizophrenia Risk Prediction
The following files are used to process data for the schizophrenia risk prediction task:
- **`race_gender_stratify_split.py`**  
  Generates the train/validation/test splits for the individuals in the schizophrenia dataset, stratified by race, gender, and outcome. Note that we use the train/validation/test splits created by the foundation model for AMI, SLE, and T2DM. 
- **`create_temporal_datasets.ipynb`**  
  Generates a flat file where each row is one "bin" for one patient containing the features used by the schizophrenia prediction model. The outputs of this are used by  **`create_dataloaders.py`** 
- **`create_causal_datasets.ipynb`**  
  Generates a nontemporal version of the schizophrenia data where there is one row per person -- rather than binning data every 90 days, we treat the entire trajectory seen by the model as one "bin" and measure frequencies over this expanded trajectory. This version of the data is used for calculating the causal effects and for any fairness interventions not compatible with the temporal data structure. 
- **`create_dataloaders.py`**  
  Generates the dataloaders used for model training. Use this file to constrict the columns for feature selection-based interventions. 
- **`generate_hcu_bysetting.py`**  
  Calculates setting-specific healthcare utilization for each patient -- this file generates setting-specific HCU (e.g., distinguishes between inpatient and outpatient). 


### Task-Specific Model: Schizophrenia Risk Prediction
- **`linprob_features.py`**  
  Pulls existing foundation model embeddings and adds in healthcare utilization/demographic information for downstream use in causal fairness pipeline. 
