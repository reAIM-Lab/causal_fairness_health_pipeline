import os
import numpy as np
import pandas as pd
from collections import Counter
import torch
from sklearn.metrics import *
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import sys
from torch.utils.data import Dataset

from models import *
from losses import *

class TabularDataset(Dataset):
    def __init__(self, df, split, feature_cols, unique_person_col = 'PID_unique', person_col = 'person_id', pred_time_col = 'prediction_time', output_col = 'boolean_value'):
        """
        Args:
            df (pd.DataFrame): The full 2D csv data.
            split (str): One of 'train', 'tuning', 'held_out'.
            feature_cols (list): List of column names to use as features.
        """
        self.data = df[df['split'] == split].reset_index(drop=True)
        
        # Ensure person_id exists for output mapping
        self.pids_unique = self.data[unique_person_col]
        self.pred_time = self.data[pred_time_col]
        self.pids = self.data[person_col].values

        self.features = self.data[feature_cols].values.astype(np.float32)
        self.labels = self.data[output_col].values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns: person_id, feature_vector, label
        return self.pids_unique[idx], self.pids[idx], self.pred_time[idx], torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

def train_lbfgs(train_loader, model, device, optimizer, criterion, loss_regularization_dict):
    model.to(device)
    model.train()

    epoch_loss = 0
    list_training_loss = []
    list_regularization_terms = []
    
    # ASSUMPTION: train_loader is Full Batch (length 1)
    for i, batch in enumerate(train_loader):
        pids_unique, pids, pred_times, features, labels = batch
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # --- Define the LBFGS Closure ---
        def closure():
            optimizer.zero_grad()
            
            # 1. Forward Pass
            outputs = model(features).squeeze()
            main_loss = criterion(outputs, labels)
            
            # 2. Add Custom Regularization INSIDE closure
            # LBFGS needs to see how changing weights effects the *entire* loss
            total_loss = main_loss
            
            if loss_regularization_dict is not None:
                regularization_term = calculate_regularization(batch, outputs, model, loss_regularization_dict)
                total_loss += regularization_term
                
            # 3. Backward Pass
            total_loss.backward()
            return total_loss

        # --- Step ---
        # LBFGS will call closure() ~2-20 times here
        loss = optimizer.step(closure)
        
        # --- Logging ---
        epoch_loss += loss.item()
        list_training_loss.append(loss.item())

        # For metrics, we do a quick no-grad pass at the end
        with torch.no_grad():
            outputs = model(features).squeeze()
            pred_ys = torch.sigmoid(outputs).detach().cpu().numpy()
            true_ys = labels.detach().cpu().numpy()
            
            # Recalculate regularization just once for logging purposes
            if loss_regularization_dict is not None:
                reg_val = calculate_regularization(batch, outputs, model, loss_regularization_dict)
                list_regularization_terms.append(reg_val.item())

    # --- Metrics Calculation (Same as before) ---
    true_ys_flattened = true_ys.ravel()
    pred_ys_flattened = pred_ys.ravel()
    pred_labels = (pred_ys_flattened > 0.5) * 1
    
    auc_train = roc_auc_score(true_ys_flattened, pred_ys_flattened)
    f1_train = f1_score(true_ys_flattened, pred_labels)
    try:
        auprc_train = average_precision_score(true_ys_flattened, pred_ys_flattened)
    except:
        auprc_train = 0
    correct_label = accuracy_score(true_ys_flattened, pred_labels)
    
    training_output_dict = {
        'auroc': auc_train, 
        'f1': f1_train, 
        'auprc': auprc_train, 
        'accuracy': correct_label, 
        'loss': np.mean(list_training_loss)
    }
    
    if len(list_regularization_terms) > 0:
        training_output_dict['regularization_loss'] = np.mean(list_regularization_terms)
    
    return training_output_dict

def train(train_loader, model, device, optimizer, criterion, loss_regularization_dict):
    model.to(device)
    model.train()

    epoch_loss = 0
    list_training_loss = []
    
    true_ys = []
    pred_ys = []
    list_regularization_terms = []
    
    for i, batch in enumerate(train_loader):
        # Unpack simplified batch (pid, features, label)
        pids_unique, pids, pred_times, features, labels = batch
        
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)

        # Optional regularization (usually None for standard LR)
        if loss_regularization_dict is not None:
            model.eval()
            regularization_term = calculate_regularization(batch, outputs, model, loss_regularization_dict)
            model.train()
            loss += regularization_term
            list_regularization_terms.append(regularization_term.item())


        epoch_loss += loss.item()
        loss.backward()
        list_training_loss.append(loss.item())
        optimizer.step()
        
        torch.cuda.synchronize()
        
        true_ys.append(labels.detach().cpu().numpy())
        # Apply sigmoid for metrics (Model outputs logits)
        pred_ys.append(torch.sigmoid(outputs).detach().cpu().numpy())

    true_ys_flattened = np.concatenate(true_ys).ravel()
    pred_ys_flattened = np.concatenate(pred_ys).ravel()
    pred_labels = (pred_ys_flattened > 0.5) * 1
    
    auc_train = roc_auc_score(true_ys_flattened, pred_ys_flattened)
    f1_train = f1_score(true_ys_flattened, pred_labels)
    try:
        auprc_train = average_precision_score(true_ys_flattened, pred_ys_flattened)
    except:
        auprc_train = 0
    correct_label = accuracy_score(true_ys_flattened, pred_labels)
    
    training_output_dict = {
        'auroc': auc_train, 
        'f1': f1_train, 
        'auprc': auprc_train, 
        'accuracy': correct_label, 
        'loss': np.mean(list_training_loss)
    }
    if len(list_regularization_terms) > 0: 
        training_output_dict['regularization_loss'] = np.mean(list_regularization_terms)
    return training_output_dict

def test(test_loader, model, device, criterion, loss_regularization_dict):
    model = model.to(device)
    model.eval()

    total_loss = 0
    list_testing_loss = []
    true_ys = []
    pred_ys = []
    list_unique_pids = []
    list_pids = []
    list_pred_times = []
    list_regularization_terms = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):    
            pids_unique, pids, pred_times, features, labels = batch
            
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            if loss_regularization_dict is not None:
                regularization_term = calculate_regularization(batch, outputs, model, loss_regularization_dict)
                loss += regularization_term
                list_regularization_terms.append(regularization_term.item())

            
            total_loss += loss.item()
            list_testing_loss.append(loss.item())

            torch.cuda.synchronize()
            
            true_ys.append(labels.detach().cpu().numpy())
            pred_ys.append(torch.sigmoid(outputs).detach().cpu().numpy())
            list_pids.append(pids)
            list_unique_pids.append(pids_unique)
            list_pred_times.append(pred_times)

    true_ys_flattened = np.concatenate(true_ys).ravel()
    pred_ys_flattened = np.concatenate(pred_ys).ravel()
    pids_flattened = np.concatenate(list_pids).ravel()
    unique_pids_flattened = np.concatenate(list_unique_pids).ravel()
    pred_times_flattened = np.concatenate(list_pred_times).ravel()
    pred_labels = (pred_ys_flattened > 0.5) * 1
    
    auc_test = roc_auc_score(true_ys_flattened, pred_ys_flattened)
    f1_test = f1_score(true_ys_flattened, pred_labels)
    try:
        auprc_test = average_precision_score(true_ys_flattened, pred_ys_flattened)
    except:
        auprc_test = 0
    correct_label = accuracy_score(true_ys_flattened, pred_labels)

    testing_output_dict = {
        'auroc': auc_test, 
        'f1': f1_test, 
        'auprc': auprc_test, 
        'accuracy': correct_label, 
        'loss': np.mean(list_testing_loss)
    }

    if len(list_regularization_terms) > 0: 
        testing_output_dict['regularization_loss'] = np.mean(list_regularization_terms)
    # Stack for CSV output
    outputs_arr = np.vstack((unique_pids_flattened, pids_flattened, pred_times_flattened, true_ys_flattened, pred_ys_flattened)).T
    return testing_output_dict, outputs_arr
    

def get_optimizer(params, optimizer_config):
    optimizer_cls = getattr(optim, optimizer_config["type"], None)
    if optimizer_cls is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")
    return optimizer_cls(params, **optimizer_config.get("params", {}))

def get_scheduler(optimizer, scheduler_config):
    scheduler_cls = getattr(lr_scheduler, scheduler_config["type"], None)
    if scheduler_cls is None:
        raise ValueError(f"Unsupported scheduler: {scheduler_config['type']}")
    return scheduler_cls(optimizer, **scheduler_config.get("params", {}))

def get_loss_function(loss_config):
    if loss_config["type"] == "WeightedBCELoss":
        return Weighted_BCELoss(weights=loss_config["weights"])
    else:
        raise ValueError(f"Unsupported loss type: {loss_config['type']}")
