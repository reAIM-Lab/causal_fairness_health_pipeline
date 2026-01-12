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

from models import *
from losses import *

def downsize_batches(loader, new_batchsize):
    """
    fix for x, z, w situation
    """
    list_pids = []
    list_xs = []
    list_zs = []
    list_ws = []
    list_padding_mask = []
    list_labels = []
    
    for pids, X, Z, W, padding_mask, labels in loader:
        list_pids.append(pids)
        list_xs.append(X)
        list_zs.append(Z)
        list_ws.append(W)
        list_padding_mask.append(padding_mask)
        list_labels.append(labels)
        
    dataset_pids = np.concatenate(list_pids)
    dataset_x = np.concatenate(list_xs)
    dataset_Z = np.concatenate(list_zs)
    dataset_W = np.concatenate(list_ws)
    dataset_padding = np.concatenate(list_padding_mask)
    dataset_labels = np.concatenate(list_labels)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(dataset_pids), torch.from_numpy(dataset_x), torch.from_numpy(dataset_Z), torch.from_numpy(dataset_W), torch.from_numpy(dataset_padding), torch.from_numpy(dataset_labels))
    smaller_test_loader = torch.utils.data.DataLoader(dataset, batch_size=new_batchsize)
    return smaller_test_loader

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

def train(train_loader, model, device, optimizer, criterion, loss_regularization_dict):
    model.to(device)
    model.train()

    epoch_loss = 0
    list_training_loss = []
    list_regularization_terms = []
    
    true_ys = []
    pred_ys = []
    for i, batch in enumerate(train_loader):
        model.train()
        pids, X, Z, W, padding_mask, labels = batch
        X = reshape_X_demo(X, num_dims = len(Z.shape), num_seq = padding_mask.shape[-1])
        signals = torch.cat((X, Z, W), dim=-1)
        
        optimizer.zero_grad()
        signals, padding_mask, labels = signals.to(device, non_blocking=True), padding_mask.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(signals, padding_mask)
        
        loss = criterion(outputs.squeeze(), labels.squeeze())

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
        pred_ys.append(outputs.detach().cpu().numpy())


    true_ys_flattened = np.concatenate(true_ys).ravel()
    pred_ys_flattened = np.concatenate(pred_ys).ravel()
    pred_labels = (pred_ys_flattened>0.5)*1
    
    auc_train = roc_auc_score(true_ys_flattened, pred_ys_flattened)
    f1_train = f1_score(true_ys_flattened, pred_labels)
    auprc_train = average_precision_score(true_ys_flattened, pred_ys_flattened)
    correct_label = accuracy_score(true_ys_flattened, pred_labels)
    
    training_output_dict = {'auroc': auc_train, 'f1': f1_train, 'auprc': auprc_train, 'accuracy': correct_label, 'loss': np.mean(list_training_loss)}
    if len(list_regularization_terms) > 0: 
        training_output_dict['regularization_loss'] = np.mean(list_regularization_terms)

    return training_output_dict

def test(test_loader, model, device, criterion, loss_regularization_dict):
    model = model.to(device)
    model.eval()

    total_loss = 0
    list_testing_loss = []
    list_regularization_terms = []
    true_ys = []
    pred_ys = []
    list_pids = []
    list_ttes = []

    for i, batch in enumerate(test_loader):    
        pids, X, Z, W, padding_mask, labels = batch
        X = reshape_X_demo(X, num_dims = len(Z.shape), num_seq = padding_mask.shape[-1])
        signals = torch.cat((X, Z, W), dim=-1)

        signals, padding_mask, labels = signals.to(device, non_blocking=True), padding_mask.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        outputs = model(signals, padding_mask)
        loss = criterion(outputs.squeeze(), labels.squeeze())
        
        if loss_regularization_dict is not None:
            regularization_term = calculate_regularization(batch, outputs, model, loss_regularization_dict)
            loss += regularization_term
            list_regularization_terms.append(regularization_term.item())

        total_loss += loss.item()
        list_testing_loss.append(loss.item())

        torch.cuda.synchronize()
        
        true_ys.append(labels.detach().cpu().numpy())
        pred_ys.append(outputs.detach().cpu().numpy())
        list_pids.append(pids.numpy())

    true_ys_flattened = np.concatenate(true_ys).ravel()
    pred_ys_flattened = np.concatenate(pred_ys).ravel()
    pids_flattened = np.concatenate(list_pids).ravel()
    pred_labels = (pred_ys_flattened>0.5)*1
    
    auc_test = roc_auc_score(true_ys_flattened, pred_ys_flattened)
    f1_test = f1_score(true_ys_flattened, pred_labels)
    auprc_test = average_precision_score(true_ys_flattened, pred_ys_flattened)
    correct_label = accuracy_score(true_ys_flattened, pred_labels)

    testing_output_dict = {'auroc': auc_test, 'f1': f1_test, 'auprc': auprc_test, 'accuracy': correct_label, 'loss': np.mean(list_testing_loss)}
    if len(list_regularization_terms) > 0: 
        testing_output_dict['regularization_loss'] = np.mean(list_regularization_terms)

    outputs_arr = np.vstack((pids_flattened, true_ys_flattened, pred_ys_flattened)).T
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

def create_grud_xmean(train_loader):
    sum_vals = None
    count_vals = None

    for batch in train_loader:
        # dataset_object = TensorDataset(pids, X, Z, W, X_mask, y_tte, y_label)
        _, X, Z, W, seq_mask, _ = batch
        Z_mask = seq_mask.unsqueeze(-1).expand(-1, -1, Z.shape[-1])
        W_mask = seq_mask.unsqueeze(-1).expand(-1, -1, W.shape[-1])
        X_mask = seq_mask.unsqueeze(-1).expand(-1, -1, X.shape[-1])

        masked_Z = Z[:,0,:,:] * Z_mask
        masked_W = W[:,0,:,:] * W_mask
        expand_X = X.unsqueeze(1).repeat(1, seq_mask.shape[-1], 1) # want overall proportions, not time-specific
        masked_temporal_feats = torch.cat((expand_X, masked_Z, masked_W), dim=-1)

        loop_sum_vals = masked_temporal_feats.sum(dim=0) # Step 3: sum over batch only where mask == 1
        loop_count_vals = seq_mask.sum(dim=0).clamp(min=1)  # avoid divide by 0

        if sum_vals is None:
            sum_vals = loop_sum_vals
            count_vals = loop_count_vals
        else:
            sum_vals += loop_sum_vals
            count_vals += loop_count_vals

    count_vals = count_vals.unsqueeze(1).repeat(1, sum_vals.shape[-1])
    mean_vals = sum_vals / count_vals        # (sequence, features)
    mean_vals = mean_vals.unsqueeze(0)
    print(mean_vals.shape)
    return mean_vals

def reshape_X_demo(X, num_dims, num_seq):
    X = X.unsqueeze(1).repeat(1, num_seq, 1)
    if num_dims == 4: 
        X = X.unsqueeze(1).repeat(1, 4, 1, 1)
        X[:, 2, :, :] = 1
        X[:, 3, :, :] = 0
    return X
    
def make_reduced_loader(loader, model, device):
    """
    Given a DataLoader whose batches are:
        pids, X, Z, W, padding_mask, labels
    Run W through the dim-reduction model and return a new loader
    that contains W_reduced in place of W.
    """
    list_pids = []
    list_xs = []
    list_zs = []
    list_ws = []
    list_padding_mask = []
    list_labels = []

    model.eval()

    with torch.no_grad():
        for batch in loader:
            pids, X, Z, W, padding_mask, labels = batch
            list_pids.append(pids)
            list_xs.append(X)
            list_zs.append(Z)
            # SKIP W
            list_padding_mask.append(padding_mask)
            list_labels.append(labels)

            # get W
            W = W.to(device)
            padding_mask = padding_mask.to(device)
            outputs = model(W, padding_mask)
            W_reduced = outputs["W"] 
            W_reduced = W_reduced.cpu()
            list_ws.append(W_reduced)

    dataset_pids = np.concatenate(list_pids)
    dataset_x = np.concatenate(list_xs)
    dataset_Z = np.concatenate(list_zs)
    dataset_W = np.concatenate(list_ws)
    dataset_padding = np.concatenate(list_padding_mask)
    dataset_labels = np.concatenate(list_labels)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(dataset_pids), torch.from_numpy(dataset_x), torch.from_numpy(dataset_Z), torch.from_numpy(dataset_W), torch.from_numpy(dataset_padding), torch.from_numpy(dataset_labels))

    # Use *exact same* batch_size/sampler/etc as original
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False)
