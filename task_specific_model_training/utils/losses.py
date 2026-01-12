import os
import numpy as np
import pandas as pd
from collections import Counter
import torch
from sklearn.metrics import *
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import sys
import itertools
import csv

class Weighted_BCELoss(nn.Module):
    def __init__(self, weights, eps=1e-6):
        super(Weighted_BCELoss, self).__init__()
        self.weights = weights
        self.eps = eps

    def forward(self, output, target, smooth=1):
        output = torch.clamp(output, self.eps, 1 - self.eps)
        loss = self.weights[1] * (target * torch.log(output)) + self.weights[0] * ((1 - target) * torch.log(1 - output))
        return -torch.mean(loss)

def calculate_regularization(batch, outputs, model, loss_regularization_dict):
    pids, X, Z, W, padding_mask, labels = batch
    if loss_regularization_dict['regularization_type'] == 'causal':
        """
        loss_regularization_dict should include:
        - x1 = race_Black
        - x0 = race_White
        - X_cols: order of columns for demographics
        - remove_col = none or list of people to remove (e.g. [race_Missing])
        - dict_effects: should look like: {'NDE': (target_value, lambda)} for each
        - relu eps
        - eps
        - device
        - px_z
        """
        c_loss = make_causal_outputs(batch, outputs, model, loss_regularization_dict)
        return c_loss
    elif loss_regularization_dict['regularization_type'] == 'eqodds':
        fpr_tpr_loss_diff = eq_odds_loss(batch, outputs, pids, loss_regularization_dict)
        return fpr_tpr_loss_diff

def make_causal_outputs(batch, outputs, model, loss_regularization_dict):
    """
    1. limit to any people with the demographics we want (e.g. not missing race)
    2. Set the demographics to whatever race/gender we want
    3. Use the model to get outputs0 and outputs1
    """
    pids, X, Z, W, padding_mask, labels = batch
    x0 = loss_regularization_dict['x0']
    x1 = loss_regularization_dict['x1']
    device = loss_regularization_dict['device']

    zero_idxs = [loss_regularization_dict['X_cols'].index(c) for c in loss_regularization_dict['remove_col']]
    mask_batch = (X[:, zero_idxs] == 0).all(dim=1)
    pids = pids[mask_batch]
    X = X[mask_batch]
    Z = Z[mask_batch]
    W = W[mask_batch]
    padding_mask = padding_mask[mask_batch]
    labels = labels[mask_batch]
    outputs = outputs[mask_batch]

    X_x0 = X.clone()
    X_x0[:, x0_idx] = 1
    X_x0[:, x1_idx] = 0

    X_x1 = X.clone()
    X_x1[:, x0_idx] = 0
    X_x1[:, x1_idx] = 1

    X_x0 = reshape_X_demo(X_x0, num_dims = len(Z.shape), num_seq = padding_mask.shape[-1])
    X_x1 = reshape_X_demo(X_x1, num_dims = len(Z.shape), num_seq = padding_mask.shape[-1])
    signals_x0 = torch.cat((X_x0, Z, W), dim=-1)
    signals_x1 = torch.cat((X_x1, Z, W), dim=-1)
    signals_x0, signals_x1, padding_mask, labels = signals_x0.to(device, non_blocking=True), signals_x1.to(device, non_blocking=True), padding_mask.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    outputs_x0 = model(signals_x0, padding_mask)
    outputs_x1 = model(signals_x1, padding_mask)

    true_X = X[:, x1_idx]
    px_z = torch.from_numpy(loss_regularization_dict['px_z'].loc[pids].values)
    
    causal_output = causal_loss(outputs, outputs_x0, outputs_x1, true_X, px_z, loss_regularization_dict['dict_effects'], loss_regularization_dict['relu_eps'], loss_regularization_dict['eps'], loss_regularization_dict['device'])
    return causal_output

    
def causal_loss(pred, pred0, pred1, X, px_z, eta_dict, relu_eps, eps, device, return_individual_losses=False):
    """
    pred = normal labels
    pred0 and pred1 = p(y|do(x))
    X = demographic variable (binary)
    px_z = propensity (p(x|z))
    eta_dict = nde, nie, spurious effects in data
    relu_eps: whether to use relu
    """

    pred_prob = torch.sigmoid(pred).squeeze()
    pred_prob1 = torch.sigmoid(pred1).squeeze()
    pred_prob0 = torch.sigmoid(pred0).squeeze()
    X_sq = X.squeeze()
    
    # get P(x | z) model
    px = X.mean()

    # get f_{x_1, W_{x_0}}
    wgh0 = ((1 - px) / (1 - px_z)).to(device)
    num_x0 = (X_sq==0).sum()
    fx1_wx0 = ((pred_prob1[X_sq == 0] * wgh0[X_sq == 0]).sum() / (wgh0[X_sq == 0].sum()))/num_x0

    # get f_{x_0, W_{x_0}}
    fx0_wx0 = ((pred_prob0[X_sq == 0] * wgh0[X_sq == 0]).sum() / (wgh0[X_sq == 0].sum()))/num_x0
    
    # get f_{x_1, W_{x_1}}
    wgh1 = (px / (px_z)).to(device)
    num_x1 = (X_sq==1).sum()
    fx1_wx1 = ((pred_prob1[X_sq == 1] * wgh1[X_sq == 1]).sum() / (wgh1[X_sq == 1].sum()))/num_x1
    
    # get f | x0
    f_x0 = pred_prob[X_sq == 0].mean()
    
    # get f | x1
    f_x1 = pred_prob[X_sq == 1].mean()
    
    # \sum_i=1^n [f(x1, w) - f(x0, w)] * 1 / n (direct effect)
    nde_loss = torch.abs(fx1_wx0 - fx0_wx0 - eta_dict['nde'][0])
    nie_loss = torch.abs(fx1_wx0 - fx1_wx1 - eta_dict['nie'][0])
    nse_loss_x1 = torch.abs(f_x1 - fx1_wx1) 
    nse_loss_x0 = torch.abs(f_x0 - fx0_wx0) # remove eta_dict because we are doing the nse loss diff
    nse_loss_diff = torch.abs((nse_loss_x1-nse_loss_x0) - eta_dict['nse'][0])
    
    # in ReLU style, penalize only larger deviations (and use larger \lambda)
    if relu_eps:
      nde_loss = torch.relu(nde_loss - eps)
      nie_loss = torch.relu(nie_loss - eps)
      nse_loss_x1 = torch.relu(nse_loss_x1 - eps)
      nse_loss_x0 = torch.relu(nse_loss_x0 - eps)
      nse_loss_diff = torch.abs(nse_loss_x1-nse_loss_x0)

    custom_loss = eta_dict['nde'][1]*nde_loss + eta_dict['nie'][1]*nie_loss + eta_dict['nse'][1]*nse_loss_diff

    if return_individual_losses == False:
        return custom_loss
    else: 
        nde_loss = (nde_loss.detach().cpu()).item()
        nie_loss = (nie_loss.detach().cpu()).item()
        nse_loss_diff = (nse_loss_diff.detach().cpu()).item()
        return nde_loss, nie_loss, nse_loss_diff

def reshape_X_demo(X, num_dims, num_seq):
    X = X.unsqueeze(1).repeat(1, num_seq, 1)
    if num_dims == 4: 
        X = X.unsqueeze(1).repeat(1, 4, 1, 1)
        X[:, 2, :, :] = 1
        X[:, 3, :, :] = 0
    return X
    
def log_causal_loss(list_metrics, log_file):
    # Create file if it doesn't exist, and write header once
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["nde_loss", "nie_loss", "expse_x0_loss", "expse_x1_loss"])  # header
        writer.writerow(list_metrics)  # append new row

def eq_odds_loss(batch, outputs, model, loss_regularization_dict):
    pids, X, _, _, _, labels = batch

    x0 = loss_regularization_dict['x0']
    x1 = loss_regularization_dict['x1']

    zero_idxs = [loss_regularization_dict['X_cols'].index(c) for c in loss_regularization_dict['remove_col']]
    mask_batch = (X[:, zero_idxs] == 0).all(dim=1)
    pids = pids[mask_batch]
    X = X[mask_batch]
    labels = labels[mask_batch]
    outputs = outputs[mask_batch]


    x0_idx = loss_regularization_dict['X_cols'].index(x0)
    idx_group_0 = X[:, x0_idx].long()
    x1_idx = loss_regularization_dict['X_cols'].index(x1)
    idx_group_1 = X[:, x1_idx].long()

    y_flat = labels.view(-1).long()
    g0_flat = idx_group_0.view(-1).long()
    g1_flat = idx_group_1.view(-1).long()

    # 3. Calculate masks safely
    # Note: Ensure you are comparing Ints to Ints. If y is float, use y_flat.long()
    mask_y0_g1 = (y_flat == 0) & (g1_flat == 1)
    mask_y1_g1 = (y_flat == 1) & (g1_flat == 1)

    mask_y0_g0 = (y_flat == 0) & (g0_flat == 1)
    mask_y1_g0 = (y_flat == 1) & (g0_flat == 1)

    # --- Calculate Soft TPR (True Positive Rate) ---
    # TPR = Sum(Preds given Label=1) / Count(Label=1)
    
    # Group 0 Positives
    if torch.sum(mask_y1_g0) > 0:
        preds_g0_pos = outputs[mask_y1_g0]
        tpr_g0 = torch.mean(preds_g0_pos)
    else:
        print('tpr g0 0')
        tpr_g0 = torch.tensor(0.0, device=outputs.device) # Handle empty batch case

    # Group 1 Positives
    if torch.sum(mask_y1_g1) > 0:
        preds_g1_pos = outputs[mask_y1_g1]
        tpr_g1 = torch.mean(preds_g1_pos)
    else:
        print('tpr g1 0')
        tpr_g1 = torch.tensor(0.0, device=outputs.device)

    # --- Calculate Soft FPR (False Positive Rate) ---
    # FPR = Sum(Preds given Label=0) / Count(Label=0)
    
    # Group 0 Negatives
    if torch.sum(mask_y0_g0) > 0:
        preds_g0_neg = outputs[mask_y0_g0]
        fpr_g0 = torch.mean(preds_g0_neg)
    else:
        print('fpr g0 0')
        fpr_g0 = torch.tensor(0.0, device=outputs.device)

    # Group 1 Negatives
    if torch.sum(mask_y0_g1) > 0:
        preds_g1_neg = outputs[mask_y0_g1]
        fpr_g1 = torch.mean(preds_g1_neg)
    else:
        print('fpr g1 0')
        fpr_g1 = torch.tensor(0.0, device=outputs.device)

    # 3. Compute Fairness Regularization Term
    # We penalize the absolute difference between groups
    diff_tpr = torch.abs(tpr_g0 - tpr_g1)
    diff_fpr = torch.abs(fpr_g0 - fpr_g1)
    print('values', tpr_g0.item(), tpr_g1.item(), fpr_g0.item(), fpr_g1.item())
    print('differences', diff_tpr.item(), diff_fpr.item())
    return loss_regularization_dict['lambda'] * (diff_tpr + diff_fpr)