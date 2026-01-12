import random
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.mixture import GaussianMixture
import pandas as pd
import pickle

class AFOExplainer:
    def __init__(self, model, data_means, activation=torch.nn.Softmax(-1)):
        self.device = torch.device("cuda:1") 
        print(self.device)
        self.base_model = model.to(self.device)
        
        # data points is the 0th slice of the grud-loader
        self.data_distribution = data_means
        print('Initial dist shapes', self.data_distribution.shape)
        self.activation = activation

    def attribute(self, X, Z, W, x_padding, ind, retrospective=False):
        """
        Compute importance score for a sample x, over time and features
        :param x: Sample instance to evaluate score for. Shape:[batch, features, time]
        :param n_samples: number of Monte-Carlo samples
        :return: Importance score matrix of shape:[batch, features, time]
        """
        _, t_len, n_features = W.shape

        # SIGNALS # REDO so no need to keep permuting stuff
        x = torch.cat((X, Z, W), dim=-1)

        model_x = x.clone()
        model_x = model_x.to(self.device)
        if retrospective:
            p_y_t = self.activation(self.base_model(model_x, x_padding)) ## change to be model_x

        ## CHANGE remove the loop of time here, since we are feeding in one timestep at a time instead of one batch of patients at a time
        if not retrospective:
            ## no longer need the :t+1 because I only have up to the given timestep in my model
            p_y_t = self.base_model(model_x, x_padding) ## change to be model_x; remove activation bc that's in my model
        score = torch.zeros((p_y_t.shape[0], n_features))
        print(score.shape)
        for i in tqdm(range(n_features)):
            # self.data distribution shape x, feats, time; use all values for feature i to create empirical distribution
            long_feature_dist = (self.data_distribution[:, i, ind]).reshape(-1)
            feature_dist = long_feature_dist[long_feature_dist!=0].to(self.device)
            if len(feature_dist) == 0:
                print(i) # make this importance 0
                score[:, i] = 0
            else:
                w_hat = W.clone()
                kl_all=[]
                for _ in range(10):
                    ## CHANGE below so we are permuting the last time point
                    rand_idx = torch.randint(0, len(feature_dist), (len(w_hat),), device=self.device)
                    w_hat[:, ind, i] = feature_dist[rand_idx]

                    x_hat = torch.cat((X, Z, w_hat), axis=-1)
                    y_hat_t = self.base_model(x_hat, x_padding) 
                    # kl = torch.nn.KLDivLoss(reduction='none')(torch.log(y_hat_t), p_y_t)
                    kl = torch.abs(y_hat_t - p_y_t)
                    # kl_all.append(torch.sum(kl, -1).cpu().detach().numpy())
                    kl_all.append(kl.detach())
                E_kl = torch.stack(kl_all).mean(dim=0)
                # score[:, i, t] = 2./(1+np.exp(-1*E_kl)) - 1.
                score[:, i] = E_kl.cpu().squeeze(-1)
        return score

