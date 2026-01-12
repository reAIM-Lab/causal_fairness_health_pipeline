import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
import os
import itertools
import copy

from autoencoder import *

def train_autoencoder(df, cols, hidden_dim, output_dim, epochs=20, batch_size=64, lr=1e-3, device="cpu"):
    """
    Trains an autoencoder on df[cols] and returns model, encoded dataframe, and loss history.
    """
    X = df[cols].values.astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = build_autoencoder(input_dim=X.shape[1], hidden_dim = hidden_dim, output_dim=output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        losses.append(epoch_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Encode the full dataset
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).to(device)
        _, encoded = model(X_tensor)
        encoded_np = encoded.cpu().numpy()

    encoded_df = pd.DataFrame(encoded_np, index=df.index,
                              columns=[f"AE_{i}" for i in range(output_dim)])
    return model, encoded_df, losses


def hyperparam_search(df, cols, output_dim, param_grid, device="cpu"):
    """
    Runs hyperparameter search over param_grid and returns best model + results.
    """
    best_loss = float("inf")
    best_params = None
    best_model = None
    best_encoded_df = None
    best_losses = None

    keys, values = zip(*param_grid.items())
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        print(f"Training with params: {params}")

        model, encoded_df, losses = train_autoencoder(
            df, cols, params['hidden_dim'], output_dim,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            lr=params["lr"],
            device=device
        )

        final_loss = losses[-1]
        print(f"Final loss = {final_loss:.4f}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = params
            best_model = copy.deepcopy(model)
            best_encoded_df = encoded_df.copy()
            best_losses = losses

    print("\nBest params:", best_params)
    return best_model, best_encoded_df, best_losses, best_params

def save_results(model, encoded_df, losses, results_path="results.json",
                 model_path="autoencoder.pt", data_path="encoded.csv"):
    """
    Saves model, encoded dataframe, and evaluation metrics.
    """
    # Save model
    torch.save(model.state_dict(), model_path)

    # Save encoded data
    encoded_df.to_csv(data_path)

    # Save loss history
    results = {"final_loss": losses[-1], "loss_history": losses}
    joblib.dump(results, results_path)

if __name__ == '__main__':
    data_prefix = 'PATH'
    model_path = data_prefix + 'PATH'
    data_path = data_prefix + 'PATH'

    device = 'cpu'

    param_grid = {"epochs": [20], 
                   "batch_size": [2048],
                   "lr": [1e-2, 1e-3, 1e-4],
                   "hidden_dim": [128, 256, 512, 1024]
                   }

    dimw_list = [750, 1000, 1250, 1500, 1750]
    reduced_dim_list = [[150, 300, 375, 450, 600], [200, 400, 500, 600, 800], [250, 500, 625, 750, 1000], [300, 600, 750, 900, 1200], [350, 700, 875, 1050, 1400]]
    for i, dimw in enumerate(dimw_list):
        if dimw < 1000: 
            dimz = 10
        else: 
            dimz = 30
        df = pd.read_csv(f'{data_prefix}/datasets_largew/full_dataset_dimw{dimw}_dimz{dimz}.csv')
        full_W_cols = [f'W{i+1}' for i in range(dimw)]
        device = 'cuda:3'
        for reduced_dim in reduced_dim_list[i]:
            print(dimw, reduced_dim)
            model, encoded_df, losses, best_params = hyperparam_search(df, full_W_cols, output_dim=reduced_dim, param_grid=param_grid, device=device)
            encoded_df = pd.concat([df, encoded_df], axis=1)
            encoded_df.drop(full_W_cols, axis=1, inplace=True)
            
            save_results(model, encoded_df, losses,
             results_path=f"{model_path}/results_dimw{dimw}_reduced_dim{reduced_dim}_dimz{dimz}.joblib",
             model_path=f"{model_path}/autoencoder_dimw{dimw}_reduced_dim{reduced_dim}_dimz{dimz}.pt",
             data_path=f"{data_path}/data_dimw{dimw}_reduced_dim{reduced_dim}_dimz{dimz}.csv")


