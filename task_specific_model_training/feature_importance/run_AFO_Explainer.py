import random
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.mixture import GaussianMixture
import pandas as pd
import pickle
import numpy as np
import sys
import re
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tnrange, tqdm_notebook
import torch.nn as nn
import time

sys.path.append('../utils')
from explainers import *
from models import *
from train_utils import downsize_batches, reshape_X_demo

seed_value = 35
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)

print(torch.get_float32_matmul_precision())

# set up dataset
"""
PATHS
"""

# load model
device = torch.device("cuda:1") 
model_config = torch.load(f"{model_directory}/best_model_config.pt", weights_only=False, map_location=device)
testing_clf = model_config['name'](**model_config['params']).to(device)
testing_clf.load_state_dict(torch.load(f"{model_directory}/best_model.pt", weights_only = False, map_location=device))
testing_clf.to(device)
# testing_clf = torch.compile(testing_clf)
testing_clf.eval()

# import data
data_dist = torch.load(f'{afo_path}/w_data_points_distribution.pt', weights_only=False)
test_loader_small = torch.load(f'{int_path}/{dataset_prefix}test_loader.pth', weights_only=False)
test_loader = downsize_batches(test_loader_small, 8192)

# AFO Explainer
explainer = AFOExplainer(testing_clf, data_dist, activation = None) 

# Run AFO
testing_ts = list(np.arange(10, 59, 10)) # testing_ts = np.asarray([59])
testing_ts.append(59)
print(testing_ts)

for ind in testing_ts:
    results_dict = {'pids':[], 'time_iter': [], 'importance_scores': [], 'ranked_feats': []}
    for i, batch in enumerate(test_loader):
        print(ind, i)
        pids, X, Z, W, padding_mask, labels = batch
        
        # limit to only people with something to change in that location
        batch_mask = padding_mask[:, ind] == 1 # get all the people who have a point at this time
        pids = pids[batch_mask]
        X = X[batch_mask]
        X = reshape_X_demo(X, num_dims = len(Z.shape), num_seq = padding_mask.shape[-1])
        Z = Z[batch_mask]
        W = W[batch_mask]
        padding_mask = padding_mask[batch_mask]
        print(pids.shape, X.shape, Z.shape, W.shape, padding_mask.shape)

        X, Z, W, padding_mask = X.to(device, non_blocking=True), Z.to(device, non_blocking=True), W.to(device, non_blocking=True), padding_mask.to(device, non_blocking=True)
        score = explainer.attribute(X, Z, W, padding_mask, ind)
        print(score.shape)
        ranked_features = np.argsort(np.argsort(-score.cpu().numpy(), axis=1), axis=1) + 1

        results_dict['pids'].append(pids)
        results_dict['time_iter'].append([ind])
        results_dict['importance_scores'].append(score)
        results_dict['ranked_feats'].append(ranked_features)

        torch.save(results_dict, f'{afo_path}/11_10_baseline_transformer_w_AFO_outputs_time{ind}.pt')
