import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, brier_score_loss

device = 'cuda:1'

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=1, dropout=0.1, binary=True):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.binary = binary

    def forward(self, X):
        return self.net(X)  # return raw logits

def brier_loss(probs, targets):
    return ((probs - targets.float())**2).mean()

def custom_loss(logits, targets):
    ce = nn.BCEWithLogitsLoss()(logits, targets.float())
    probs = torch.softmax(logits, dim=1)
    brier = brier_loss(probs, targets)
    return ce + 0.5*brier   # mix (even)


class TorchNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.1,
                 lr=1e-3, batch_size=1024, max_epochs=20, device=device, 
                 weight_decay = 0, label_smoothing = 0):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.device = device
        self.weight_decay = weight_decay
        self._model = None
        self.label_smoothing = label_smoothing

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)

        self._model = MLP(self.input_dim, self.hidden_dims, output_dim=1, dropout=self.dropout, binary=True).to(self.device)
        criterion = custom_loss
        optimizer = optim.Adam(self._model.parameters(), lr=self.lr)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        for epoch in range(self.max_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        self._model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self._model(X)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.hstack([1 - probs, probs])  # shape (n,2)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1])
    
    def score(self, X, y):
        # I want score to be negative brier
        y_pred = self.predict(X)
        return -brier_score_loss(y, y_pred)


class MLPRegressorNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))   # regression output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)   # shape (batch,)


# ----- sklearn-style Wrapper -----
class TorchNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=None, hidden_dims=(64, 32), dropout=0.1, 
                 lr=1e-3, batch_size=1024, max_epochs=20, device="cuda:1"):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.device = device

        self._model = None

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model = MLPRegressorNet(self.input_dim or X.shape[1], 
                                      hidden_dims=self.hidden_dims, 
                                      dropout=self.dropout).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self._model.parameters(), lr=self.lr)

        self._model.train()
        for epoch in range(self.max_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self._model(xb)
                loss = criterion(preds, yb.view(-1))
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self._model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self._model(X)
        return preds.cpu().numpy()

def grid_search_nn(X, Y, device, binary, hparams):
    hparams = {
    'lr': [1e-3, 1e-4],
    'max_epochs': [20],
    'hidden_dims': [[16, 8], [32, 16], [64, 32], [128,64]],
    'dropout': [0.1, 0.3]}


    input_dim = X.shape[1]
    if binary:
        model = TorchNNClassifier(input_dim=input_dim, device=device)
        scoring = "neg_brier_score"
    else:
        model = TorchNNRegressor(input_dim=X.shape[1], device=device)
        scoring = "neg_mean_squared_error"

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=hparams,
        scoring=scoring,
        cv=5,
        verbose=0,
    )
    X = X.values.astype('float32')
    Y = Y.values.astype('float32').reshape((len(Y), 1))
    grid_search.fit(X, Y)
    return grid_search.best_params_


