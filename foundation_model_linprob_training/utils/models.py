import numpy as np
import os
import pandas as pd
from collections import Counter
import sys
import gc
from scipy.sparse import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle 
import random
import math
from joblib import dump, load
import torch.utils.data as utils
from torch.nn.parameter import Parameter
import time


class Weighted_BCELoss(nn.Module):
    def __init__(self, weights, eps=1e-6):
        super(Weighted_BCELoss, self).__init__()
        self.weights = weights
        self.eps = eps

    def forward(self, output, target, smooth=1):
        output = torch.clamp(output, self.eps, 1 - self.eps)
        loss = self.weights[1] * (target * torch.log(output)) + self.weights[0] * ((1 - target) * torch.log(1 - output))
        return -torch.mean(loss)

class TabularTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        # 1️⃣ Embed each scalar feature into d_model
        self.feature_embedding = nn.Linear(1, d_model)

        # 2️⃣ Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 3️⃣ Output head
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (batch, num_features)
        returns: (batch, 1) probability
        """

        # (B, F) → (B, F, 1)
        x = x.unsqueeze(-1)

        # (B, F, 1) → (B, F, d_model)
        x = self.feature_embedding(x)

        # Transformer encoder
        x = self.encoder(x)

        # Pool across features
        x = x.mean(dim=1)  # (B, d_model)

        # Output probability
        y = torch.sigmoid(self.classifier(x))
        return y

class TabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims=(128, 64),
        dropout: float = 0.1,
        prior_prob: float = None  # New optional argument
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        self.encoder = nn.Sequential(*layers)

        # Output head
        self.output = nn.Linear(prev_dim, 1)

        # Initialize bias if prior_prob is provided
        if prior_prob is not None:
            if not (0 < prior_prob < 1):
                raise ValueError("prior_prob must be between 0 and 1")
            
            # Formula: b = log( p / (1-p) )
            bias_init = math.log(prior_prob / (1 - prior_prob))
            self.output.bias.data.fill_(bias_init)

    def forward(self, x):
        """
        x: (batch, input_dim)
        returns: (batch, 1) probability
        """
        h = self.encoder(x)
        y = torch.sigmoid(self.output(h))
        return y

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, prior_prob: float = None): # New optional argument
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

        # Initialize bias if prior_prob is provided
        if prior_prob is not None:
            if not (0 < prior_prob < 1):
                raise ValueError("prior_prob must be between 0 and 1")
                
            bias_init = math.log(prior_prob / (1 - prior_prob))
            self.linear.bias.data.fill_(bias_init)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))