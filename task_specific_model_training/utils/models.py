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
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:  # Handle the case when d_model is odd
            pe[:, 0, 1::2] = torch.cos(position * div_term[:-1])  # Cos for odd indices
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)  # Cos for odd indices


        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModelNoPE(nn.Module):
    def __init__(self, hidden_size, dim_feedforward, num_layers, num_heads, dropout, n_features):
        super(TransformerModelNoPE, self).__init__()
        self.embedding = nn.Linear(n_features, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward, dropout), num_layers)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, padding_mask): 
        x = self.embedding(x) # embedding shape batch, seq_len, n_feats
        
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)# transformer shape should be seq_len, batch, hidden size
        x = x*padding_mask.T.unsqueeze(-1) # now mask based on padding
        
        # getting predictions for all time points
        x = x[-1, :, :]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class TransformerModelPEEmb(nn.Module):
    def __init__(self, hidden_size, dim_feedforward, num_layers, num_heads, dropout, n_features):
        super(TransformerModelPEEmb, self).__init__()
        self.pe = PositionalEncoding(d_model = n_features, dropout = 0.0)
        self.embedding = nn.Linear(n_features, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward, dropout), num_layers)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, padding_mask): 
        x = x.permute(1, 0, 2)
        x = self.pe(x) # pe shape should be seq_len, batch_size, n_feature
        
        x = x.permute(1, 0, 2) 
        x = self.embedding(x) # embedding shape batch, seq_len, n_feats
        
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)# transformer shape should be seq_len, batch, hidden size
        x = x*padding_mask.T.unsqueeze(-1) # now mask based on padding
        
        # getting predictions for all time points
        x = x[-1, :, :]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
class TransformerModelEmbPE(nn.Module):
    def __init__(self, hidden_size, dim_feedforward, num_layers, num_heads, dropout, n_features):
        super(TransformerModelEmbPE, self).__init__()
        self.pe = PositionalEncoding(d_model = hidden_size, dropout = 0.0)
        self.embedding = nn.Linear(n_features, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward, dropout), num_layers)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, padding_mask): 
        x = self.embedding(x) # embedding shape batch, seq_len, n_feats
        
        x = x.permute(1, 0, 2)
        x = self.pe(x) # pe shape should be seq_len, batch_size, hidden_size
        x = self.transformer_encoder(x)# transformer shape should be seq_len, batch, hidden size
        x = x*padding_mask.T.unsqueeze(-1) # now mask based on padding

        # getting predictions for last time point
        x = x[-1, :, :]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, device, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        
        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.to(self.device), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)
        
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
#         print(self.weight.data)
#         print(self.bias.data)

    def forward(self, input):
#         print(self.filter_square_matrix.mul(self.weight))
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias).to(self.device)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'
        
class GRUD(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size, X_mean, device, output_last = False):
        """
        Recurrent Neural Networks for Multivariate Times Series with Missing Values
        GRU-D: GRU exploit two representations of informative missingness patterns, i.e., masking and time interval.
        cell_size is the size of cell_state.
        
        Implemented based on the paper: 
        @article{che2018recurrent,
          title={Recurrent neural networks for multivariate time series with missing values},
          author={Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan},
          journal={Scientific reports},
          volume={8},
          number={1},
          pages={6085},
          year={2018},
          publisher={Nature Publishing Group}
        }
        
        GRU-D:
            input_size: variable dimension of each time
            hidden_size: dimension of hidden_state
            mask_size: dimension of masking vector
            X_mean: the mean of the historical input data
        """
        
        super(GRUD, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size

        self.identity = torch.eye(input_size).to(self.device)
        self.zeros = Variable(torch.zeros(input_size).to(self.device))
        self.X_mean = Variable(torch.Tensor(X_mean).to(self.device))
        
        self.zl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.rl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.hl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        
        self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, self.identity, self.device).to(self.device)
        
        self.gamma_h_l = nn.Linear(self.delta_size, self.delta_size).to(self.device)
        
        self.output_last = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
        
    def step(self, x, x_last_obsv, x_mean, h, mask, delta):
        
        batch_size = x.shape[0]
        dim_size = x.shape[1]
        delta = delta.to(self.device)

        delta_x = torch.exp(-torch.max(self.zeros, self.gamma_x_l(delta))).to(self.device)
        delta_h = torch.exp(-torch.max(self.zeros, self.gamma_h_l(delta))).to(self.device)
        
        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)
        h = delta_h * h
        
        combined = torch.cat((x, h, mask), 1)
        
        z = F.sigmoid(self.zl(combined))
        r = F.sigmoid(self.rl(combined))
        combined_r = torch.cat((x, r * h, mask), 1)
        h_tilde = F.tanh(self.hl(combined_r))
        h = (1 - z) * h + z * h_tilde
        
        return h
    
    def forward(self, input, padding_mask):
        batch_size = input.size(0)
        type_size = input.size(1)
        step_size = input.size(2)
        spatial_size = input.size(3)
        
        Hidden_State = self.initHidden(batch_size).to(self.device)
        X = torch.squeeze(input[:,0,:,:]).to(self.device)
        X_last_obsv = torch.squeeze(input[:,1,:,:]).to(self.device)
        Mask = torch.squeeze(input[:,2,:,:]).to(self.device)
        Delta = torch.squeeze(input[:,3,:,:]).to(self.device)
        
        outputs = None
        for i in range(step_size):
            Hidden_State = self.step(torch.squeeze(X[:,i:i+1,:])\
                                     , torch.squeeze(X_last_obsv[:,i:i+1,:])\
                                     , torch.squeeze(self.X_mean[:,i:i+1,:])\
                                     , Hidden_State\
                                     , torch.squeeze(Mask[:,i:i+1,:])\
                                     , torch.squeeze(Delta[:,i:i+1,:])).to(self.device)
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
        # Add padding
        outputs = outputs*padding_mask.unsqueeze(-1) # add a mask to zero out the invalid (padded) OUT
        outputs = outputs[:,-1,:] # get final layer
        outputs = self.output_last(outputs)
        outputs = self.sigmoid(outputs)
        return outputs
    
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State


