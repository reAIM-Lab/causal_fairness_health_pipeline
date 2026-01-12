import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from aif360.datasets import BinaryLabelDataset

class LFRModule(nn.Module):
    """
    The core PyTorch module for LFR.
    """
    def __init__(self, num_features, k=10):
        super().__init__()
        # V: Prototypes (k x num_features)
        # We initialize them randomly from a normal distribution
        self.prototypes = nn.Parameter(torch.randn(k, num_features))
        
        # W: Classification weights (k x 1) - mapping prototypes to labels
        self.w = nn.Parameter(torch.randn(k, 1))
        
        # Gamma: Scale parameter for softmax (optional, helps convergence)
        # We can fix this or learn it. Zemel paper often implies a fixed scale, 
        # but learning it adds flexibility. Here we fix it to 1.0 for stability.
        self.gamma = 1.0 

    def get_M(self, x):
        """
        Compute M: The probability of each sample x belonging to each prototype k.
        M_nk = exp(-gamma * ||x_n - v_k||^2) / Sum_j(...)
        """
        # x: (N, features)
        # prototypes: (k, features)
        
        # Efficient L2 distance calculation: ||a-b||^2 = a^2 + b^2 - 2ab
        x_sq = torch.sum(x**2, dim=1, keepdim=True)       # (N, 1)
        v_sq = torch.sum(self.prototypes**2, dim=1)       # (k)
        prod = torch.matmul(x, self.prototypes.t())       # (N, k)
        
        dists = x_sq + v_sq - 2 * prod                    # (N, k)
        
        # Softmax over the prototype dimension (dim=1)
        # We use negative distance because closer = higher probability
        M = torch.softmax(-self.gamma * dists, dim=1)
        return M

    def forward(self, x):
        M = self.get_M(x)
        
        # Reconstruct X: (N, k) @ (k, features) -> (N, features)
        x_hat = torch.matmul(M, self.prototypes)
        
        # Predict Y: (N, k) @ (k, 1) -> (N, 1)
        y_hat = torch.matmul(M, self.w)
        
        return x_hat, y_hat, M

class LFR_GPU:
    """
    AIF360-compatible wrapper for the PyTorch LFR model.
    """
    def __init__(self, unprivileged_groups, privileged_groups, 
                 k=5, Ax=0.01, Ay=1.0, Az=50.0, 
                 learning_rate=1e-3, epochs=1000, batch_size=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 verbose=0):
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        self.k = k
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size # If None, uses full batch (recommended for L_z)
        self.device = device
        self.verbose = verbose
        self.model = None

    def fit(self, dataset):
        # 1. Prepare Data
        df, _ = dataset.convert_to_dataframe()
        
        # Extract features and targets
        x_np = dataset.features
        y_np = dataset.labels.flatten()
        
        # Identify sensitive indices for L_z calculation
        # AIF360 datasets can be complex, but usually we just look at the protected attribute column
        # We need a boolean mask for unprivileged/privileged
        # (Assuming single protected attribute for simplicity based on your snippets)
        prot_attr = list(self.privileged_groups[0].keys())[0]
        priv_val = self.privileged_groups[0][prot_attr]
        
        # Get the index of the protected attribute in the feature matrix
        # (Or extract it from metadata if it's not in features)
        try:
            prot_idx = dataset.feature_names.index(prot_attr)
            s_np = x_np[:, prot_idx]
        except ValueError:
            # If protected attribute is not in features, we must rely on dataset.protected_attributes
            # This is safer for AIF360 datasets
            s_np = dataset.protected_attributes.flatten()

        is_priv = (s_np == priv_val).astype(float)
        
        # To Tensor
        x_tensor = torch.tensor(x_np, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_np, dtype=torch.float32).view(-1, 1).to(self.device)
        priv_mask = torch.tensor(is_priv, dtype=torch.float32).view(-1, 1).to(self.device)
        
        # 2. Initialize Model
        num_features = x_tensor.shape[1]
        self.model = LFRModule(num_features, k=self.k).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 3. Training Loop
        # Note: LFR typically requires Full Batch for stable L_z (Statistical Parity) calculation.
        # If dataset is massive (>100k), you might need mini-batches, but L_z becomes noisy.
        
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            
            x_hat, y_hat, M = self.model(x_tensor)
            
            # --- Losses ---
            
            # L_x: Reconstruction Error (MSE)
            loss_x = torch.mean((x_tensor - x_hat) ** 2)
            
            # L_y: Prediction Error (BCE)
            # Use BCEWithLogits if y_hat were logits, but LFR usually formulates y_hat as probability sum.
            # We clip y_hat to (0,1) for safety or use BCE with logits formulation.
            # Original paper formulation: -Sum(y log y_hat + (1-y) log(1-y_hat))
            # For numerical stability in PyTorch, we can use a sigmoid on y_hat inside the BCE.
            # However, LFR definition of y_hat is linear comb of W. Let's pass through Sigmoid.
            y_prob = torch.sigmoid(y_hat) 
            loss_y = nn.BCELoss()(y_prob, y_tensor)
            
            # L_z: Fairness (Statistical Parity)
            # M_nk is probability sample n is in prototype k
            # We want Avg(M_k | Privileged) == Avg(M_k | Unprivileged)
            
            # Mean assignment for Privileged
            M_priv = torch.sum(M * priv_mask, dim=0) / (torch.sum(priv_mask) + 1e-6)
            # Mean assignment for Unprivileged
            M_unpriv = torch.sum(M * (1-priv_mask), dim=0) / (torch.sum(1-priv_mask) + 1e-6)
            
            # Sum of absolute differences over all k
            loss_z = torch.sum(torch.abs(M_priv - M_unpriv))
            
            # Total Loss
            loss = (self.Ax * loss_x) + (self.Ay * loss_y) + (self.Az * loss_z)
            
            loss.backward()
            optimizer.step()
            
            if self.verbose > 0 and epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f} (Lx: {loss_x:.4f}, Ly: {loss_y:.4f}, Lz: {loss_z:.4f})")
                
        return self

    def transform(self, dataset):
        """
        Returns a new dataset where 'features' are the latent representations (M).
        """
        self.model.eval()
        x_np = dataset.features
        x_tensor = torch.tensor(x_np, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            M = self.model.get_M(x_tensor)
            
        # Create copy of dataset
        dataset_new = dataset.copy()
        # Replace features with the Latent Prototypes Probabilities
        dataset_new.features = M.cpu().numpy()

        new_feature_names = [f"latent_proto_{i}" for i in range(M.cpu().numpy().shape[1])]
        dataset_new.feature_names = new_feature_names
        return dataset_new

    def predict(self, dataset):
        """
        Returns a new dataset with 'labels' updated to the model predictions.
        """
        self.model.eval()
        x_np = dataset.features
        x_tensor = torch.tensor(x_np, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            _, y_hat, _ = self.model(x_tensor)
            y_prob = torch.sigmoid(y_hat).cpu().numpy()
            
        dataset_new = dataset.copy()

        dataset_new.scores = y_prob
        dataset_new.labels = (y_prob > 0.5).astype(float)
        return dataset_new