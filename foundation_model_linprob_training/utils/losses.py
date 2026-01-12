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
    pids_unique, pids, pred_times, features, labels = batch
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
    pids_unique, pids, pred_times, features, labels = batch
    x0 = loss_regularization_dict['x0']
    x1 = loss_regularization_dict['x1']
    device = loss_regularization_dict['device']
        
    mask_batch = get_mask_max_either(loss_regularization_dict, features)
    pids_unique = pids_unique[mask_batch]
    features = features[mask_batch]
    labels = labels[mask_batch]
    outputs = outputs[mask_batch]

    x0_idx = loss_regularization_dict['X_cols'].index(x0)
    x1_idx = loss_regularization_dict['X_cols'].index(x1)

    signals_x0 = features.clone()
    signals_x0[:, x0_idx] = features[:, x0_idx].max()
    signals_x0[:, x1_idx] = features[:, x1_idx].max()

    signals_x1 = features.clone()
    signals_x1[:, x0_idx] = features[:, x0_idx].min()
    signals_x1[:, x1_idx] = features[:, x1_idx].max()

    signals_x0, signals_x1, labels = signals_x0.to(device, non_blocking=True), signals_x1.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    outputs_x0 = model(signals_x0)
    outputs_x1 = model(signals_x1)

    true_X = features[:, x1_idx]
    px_z = torch.from_numpy(loss_regularization_dict['px_z'].loc[pids_unique].values)
    
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
    # set X_sq to be binary
    X_sq[X_sq==X_sq.min()] = 0
    X_sq[X_sq==X_sq.max()] = 1
    
    # get P(x | z) model
    px = X.mean()

    # get f_{x_1, W_{x_0}}
    wgh0 = ((1 - px) / (1 - px_z)).to(device)
    num_x0 = (X_sq==X_sq.min()).sum()
    fx1_wx0 = ((pred_prob1[X_sq == 0] * wgh0[X_sq == 0]).sum() / (wgh0[X_sq == 0].sum()))/num_x0
    # get f_{x_0, W_{x_0}}
    fx0_wx0 = ((pred_prob0[X_sq == 0] * wgh0[X_sq == 0]).sum() / (wgh0[X_sq == 0].sum()))/num_x0
    
    # get f_{x_1, W_{x_1}}
    wgh1 = (px / (px_z)).to(device)
    num_x1 = (X_sq==X_sq.max()).sum()
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


def get_mask(loss_regularization_dict, features, atol=1e-5):
    """
    Create a mask that keeps rows where all columns listed in
    loss_regularization_dict['remove_col'] are equal to the minimum
    value of that column (computed over the batch).

    Args
    ----
    loss_regularization_dict : dict
        Must contain:
            - 'X_cols': list of all feature column names (length D)
            - 'remove_col': list of column names to constrain
    features : torch.Tensor
        Shape (N, D)
    atol : float, optional
        Tolerance for float comparison (use >0 for float features)

    Returns
    -------
    mask : torch.BoolTensor
        Shape (N,)
    """

    # Nothing to mask → keep everything
    if (
        loss_regularization_dict is None
        or "remove_col" not in loss_regularization_dict
        or len(loss_regularization_dict["remove_col"]) == 0
    ):
        return torch.ones(features.size(0), dtype=torch.bool, device=features.device)

    X_cols = loss_regularization_dict["X_cols"]
    remove_cols = loss_regularization_dict["remove_col"]

    # Column name → index
    col_to_idx = {c: i for i, c in enumerate(X_cols)}
    remove_idx = torch.tensor(
        [col_to_idx[c] for c in remove_cols],
        device=features.device
    )

    # Extract relevant columns: (N, K)
    remove_features = features[:, remove_idx]

    # Column-wise minimum: (K,)
    col_mins = remove_features.min(dim=0).values

    # Equality check
    if atol > 0:
        equal_to_min = torch.isclose(remove_features, col_mins, atol=atol)
    else:
        equal_to_min = remove_features == col_mins

    # Row-wise AND → (N,)
    mask = equal_to_min.all(dim=1)

    return mask
def get_mask_max_either(loss_regularization_dict, features, atol=1e-5):
    """
    Keep rows where the value is equal to the column-wise maximum
    in at least ONE of the specified columns.

    Required in loss_regularization_dict:
        - 'X_cols': list of all column names
        - 'max_either_cols': list of TWO column names

    Args
    ----
    features : torch.Tensor
        Shape (N, D)
    atol : float
        Tolerance for float comparisons

    Returns
    -------
    mask : torch.BoolTensor
        Shape (N,)
    """

    if (
        loss_regularization_dict is None
        or "max_either_cols" not in loss_regularization_dict
        or len(loss_regularization_dict["max_either_cols"]) != 2
    ):
        raise ValueError("max_either_cols must contain exactly two column names")

    X_cols = loss_regularization_dict["X_cols"]
    c1, c2 = loss_regularization_dict["max_either_cols"]

    # Column indices
    col_to_idx = {c: i for i, c in enumerate(X_cols)}
    idx1, idx2 = col_to_idx[c1], col_to_idx[c2]

    # Extract columns
    col1 = features[:, idx1]
    col2 = features[:, idx2]

    # Column-wise max
    max1 = col1.max()
    max2 = col2.max()

    # Equality checks
    if atol > 0:
        is_max1 = torch.isclose(col1, max1, atol=atol)
        is_max2 = torch.isclose(col2, max2, atol=atol)
    else:
        is_max1 = col1 == max1
        is_max2 = col2 == max2

    # OR condition
    mask = is_max1 | is_max2

    return mask

def eq_odds_loss(batch, outputs, model, loss_regularization_dict):
    pids_unique, pids, pred_times, features, labels = batch
    x0 = loss_regularization_dict['x0']
    x1 = loss_regularization_dict['x1']
    x0_idx = loss_regularization_dict['X_cols'].index(x0)
    x1_idx = loss_regularization_dict['X_cols'].index(x1)
        
    mask_batch = get_mask_max_either(loss_regularization_dict, features)
    pids_unique = pids_unique[mask_batch]
    features = features[mask_batch]
    labels = labels[mask_batch]
    outputs = outputs[mask_batch]

    # make binary idx_groups 
    idx_group_0 = features[:, x0_idx]
    idx_group_0[idx_group_0==idx_group_0.min()] = 0
    idx_group_0[idx_group_0==idx_group_0.max()] = 1
    idx_group_0 = idx_group_0.long()

    idx_group_1 = features[:, x1_idx]
    idx_group_1[idx_group_1==idx_group_1.min()] = 0
    idx_group_1[idx_group_1==idx_group_1.max()] = 1
    idx_group_1 = idx_group_1.long()

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
    # print('values', tpr_g0.item(), tpr_g1.item(), fpr_g0.item(), fpr_g1.item())
    # print('differences', diff_tpr.item(), diff_fpr.item())
    return loss_regularization_dict['lambda'] * (diff_tpr + diff_fpr)