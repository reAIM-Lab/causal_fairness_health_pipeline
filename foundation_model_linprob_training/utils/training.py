import torch
import torch.nn as nn
import torch.optim as optim
import copy
import json
from pathlib import Path
from collections import defaultdict
import sys
import pandas as pd

from models import *
from losses import *
from train_utils import *

def train_model(train_loader, val_loader, model_config, config, device):
    model = model_config['name'](**model_config['params'])
    loss_config = config["loss"]
    loss_config['weights'] = torch.Tensor(loss_config['weights']).to(device)
    criterion = get_loss_function(loss_config)
    optimizer = get_optimizer(model.parameters(), config["optimizer"])
    scheduler = get_scheduler(optimizer, config["scheduler"])
    early_stopper = EarlyStopping(patience=config["early_stopping_patience"])

    if config['loss_regularization_dict'] == None:
        history = {"epoch":[], "train_loss": [], "val_loss": [], 'train_auroc':[], "val_auroc": []}
    else:
        history = {"epoch":[], "train_loss": [], "val_loss": [], 'train_auroc':[], "val_auroc": [], 'train_regularization_loss':[], 'val_regularization_loss':[]} 
    
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(config["epochs"]):
        train_metrics = train(train_loader, model, device, optimizer, criterion, config.get("loss_regularization_dict"))
        val_metrics, _ = test(val_loader, model, device, criterion, config.get("loss_regularization_dict"))
        print(f'Epoch {epoch}: Validation Loss: {val_metrics["loss"]}; AUROC: {val_metrics["auroc"]}')

        history['epoch'].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_auroc"].append(train_metrics["auroc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_auroc"].append(val_metrics["auroc"])
        
        if 'regularization_loss' in train_metrics.keys():
            history['train_regularization_loss'].append(train_metrics['regularization_loss'])
            history['val_regularization_loss'].append(val_metrics['regularization_loss'])


        scheduler.step()

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model_state = copy.deepcopy(model.state_dict())

        early_stopper(val_metrics['loss'], model)
        if early_stopper.early_stop:
            print("Early stopping")
            break
    return best_model_state, history, best_val_loss


def search_hyperparams(hparam_configs, model_configs, train_loader, val_loader, test_loader, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    results_file = os.path.join(save_dir, "all_results.csv")

    save_dir_allconfigs = f'{save_dir}/allconfig_history/'
    os.makedirs(save_dir_allconfigs, exist_ok=True)

    # Load existing results if resuming
    if os.path.exists(results_file):
        past_results = pd.read_csv(results_file)
        completed_indices = set(past_results["config_idx"].tolist())
        best_loss = past_results["val_loss"].min()
    else:
        past_results = pd.DataFrame()
        completed_indices = set()
        best_loss = float('inf')

    best_model_data = {}

    for i, config in enumerate(hparam_configs):
        loss_dict_key = config['model_selection'] # loss or auroc
        if i in completed_indices:
            print(f"Skipping config {i+1}/{len(hparam_configs)} (already done)")
            continue

        print(f"Running config {i+1}/{len(hparam_configs)}")
        model_config = model_configs[i]
        print(model_config)
        model_state, history, val_loss = train_model(train_loader, val_loader, model_config, config, device)
        
        pd.DataFrame(history).to_csv(os.path.join(save_dir_allconfigs, f"training_history_{i}.csv"))
        
        # get loss for if there is no weighting on the BCE -- "fair" comparison in hyperparam search
        val_loss_config = config["loss"]
        val_criterion = get_loss_function(val_loss_config)
        model = model_config['name'](**model_config['params'])
        model.load_state_dict(model_state)
        val_loss_dict, val_output = test(val_loader, model, device, val_criterion, 
                                         config.get("loss_regularization_dict"))
                                         
        val_loss = val_loss_dict[loss_dict_key]
        if loss_dict_key == 'auroc': # negative so we minimize still
            val_loss = -val_loss
        print('New loss', val_loss)
        print('Best loss', best_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_data = {
                "state_dict": model_state,
                "hparam_config": config,
                'model_config': model_config,
                "history": history
            }

            # Save best model and outputs
            torch.save(best_model_data["state_dict"], os.path.join(save_dir, "best_model.pt"))
            torch.save(best_model_data["hparam_config"], os.path.join(save_dir, "best_hparam_config.pt"))
            torch.save(best_model_data["model_config"], os.path.join(save_dir, "best_model_config.pt"))
            pd.DataFrame(best_model_data["history"]).to_csv(os.path.join(save_dir, "training_history.csv"))

        # ---- Save metrics + config ----
        row = {
            "config_idx": i,
            "val_loss": val_loss,
        }
        results_df = pd.DataFrame([row])
        results_df.to_csv(results_file, mode="a", header=not os.path.exists(results_file), index=False)
