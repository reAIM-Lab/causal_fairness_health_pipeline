
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

    
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


class MLP_AutoencoderModel(nn.Module):
    def __init__(self, input_dim, reduced_dim, hidden_dim=512,
                 predict_x=None, predict_y=True):
        super().__init__()

        self.reduced_dim = reduced_dim
        self.predict_X = predict_x is not None
        self.predict_Y = predict_y

        # ----------------------
        # Encoder: per-token MLP
        # ----------------------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, reduced_dim),
        )

        # ----------------------
        # Decoder (reconstruction)
        # ----------------------
        self.reconstruct_W = nn.Sequential(
            nn.Linear(reduced_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # ----------------------
        # Optional X head
        # ----------------------
        if self.predict_X:
            self.x_output_dim = predict_x["x_output_dim"]
            self.fc_x = nn.Sequential(
                nn.Linear(reduced_dim, reduced_dim // 2),
                nn.ReLU(),
                nn.Linear(reduced_dim // 2, self.x_output_dim)
            )

        # ----------------------
        # Optional Y head
        # ----------------------
        if self.predict_Y:
            self.fc_y = nn.Sequential(
                nn.Linear(reduced_dim, reduced_dim // 2),
                nn.ReLU(),
                nn.Linear(reduced_dim // 2, 1)
            )

    def forward(self, W, padding_mask):
        """
        W: (batch, seq, input_dim)
        padding_mask: (batch, seq)
        """

        # Encode each timestep independently
        encoded = self.encoder(W)                       # (B, S, reduced_dim)
        encoded = encoded * padding_mask.unsqueeze(-1)  # mask padded positions
        outputs = {"W": encoded}

        # Reconstruct W
        W_recon = self.reconstruct_W(encoded)
        outputs["W_recon"] = W_recon

        # Pooling for X/Y prediction
        encoded_pooled = encoded.mean(dim=1)            # (B, reduced_dim)

        if self.predict_X:
            logits_x = self.fc_x(encoded_pooled)
            if self.x_output_dim == 1:
                outputs["X_pred"] = torch.sigmoid(logits_x)
            else:
                outputs["X_pred"] = F.softmax(logits_x, dim=-1)

        if self.predict_Y:
            logits_y = self.fc_y(encoded_pooled)
            outputs["Y_pred"] = torch.sigmoid(logits_y)

        return outputs


class SimpleTransformer_AutoencoderModel(nn.Module):
    def __init__(
        self,
        input_dim,            # raw input feature dimension
        d_model,              # internal Transformer dimension
        output_dim,      # encoder output size (not tied to d_model)
        num_layers=4,       
        num_heads=4,
        dropout=0.1,
        predict_x=None,
        predict_y=True
    ):
        super().__init__()

        self.predict_X = predict_x is not None
        self.predict_Y = predict_y

        # ---- Input projection: raw features -> d_model ----
        self.input_proj = nn.Linear(input_dim, d_model)

        # ---- Positional encoding ----
        self.pe = PositionalEncoding(d_model=d_model, dropout=0.0)

        # ---- Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # ---- Output projection: d_model -> output_dim ----
        # (output_dim is your reduced_dim)
        self.output_dim = output_dim if output_dim is not None else d_model
        self.output_proj = nn.Linear(d_model, self.output_dim)

        # ---- Reconstruction head (output_dim -> input_dim)
        self.reconstruct_W = nn.Linear(self.output_dim, input_dim)

        # ---- Optional X head ----
        if self.predict_X:
            self.x_output_dim = predict_x["x_output_dim"]
            self.fc_x = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.output_dim // 2, self.x_output_dim)
            )

        # ---- Optional Y head ----
        if self.predict_Y:
            self.fc_y = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.output_dim // 2, 1)
            )

    def forward(self, W, padding_mask):
        """
        W: (batch, seq, input_dim)
        padding_mask: (batch, seq)
        """

        # ---- Project raw input to transformer dimension ----
        x = self.input_proj(W)   # (batch, seq, d_model)

        # ---- Apply positional encoding ----
        x = self.pe(x.transpose(0,1)).transpose(0,1)

        # ---- Transformer encoder ----
        x = self.encoder(x, src_key_padding_mask=(padding_mask == 0))

        # ---- Project encoded representation to output_dim ----
        encoded = self.output_proj(x)

        # ---- Reconstruction ----
        W_recon = self.reconstruct_W(encoded)
        outputs = {"W": encoded}
        outputs["W_recon"] = W_recon

        # ---- Mean pool for X/Y prediction ----
        pooled = encoded.mean(dim=1)

        if self.predict_X:
            logits_x = self.fc_x(pooled)
            if self.x_output_dim == 1:
                outputs["X_pred"] = torch.sigmoid(logits_x)
            else:
                outputs["X_pred"] = F.softmax(logits_x, dim=-1)

        if self.predict_Y:
            logits_y = self.fc_y(pooled)
            outputs["Y_pred"] = torch.sigmoid(logits_y)

        return outputs