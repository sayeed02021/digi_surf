import torch
import torch.nn as nn
import torch_geometric.nn as nng
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

class RegressionHead(nn.Module):
    def __init__(self, n_prop, dim):
        super().__init__()
        self.dim = dim

        self.layers = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.dim),
            nn.Linear(self.dim, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,n_prop)
        )

    def forward(self, x):
        return self.layers(x)

class AttentiveFPModel(nn.Module):
    def __init__(
            self, 
            props,
            args
    ):
        super().__init__()
        self.props = props
        self.n_prop = len(props)
        self.hidden_channels = args.hidden_channels
        self.out_channels = args.out_channels
        self.num_layers = args.num_layers
        self.num_timesteps = args.num_timesteps
        self.dropout = args.dropout
        self.in_channels = args.in_channels
        self.edge_dim = args.edge_dim

        self.encoder = nng.AttentiveFP(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            edge_dim=self.edge_dim,
            num_layers=self.num_layers,
            num_timesteps=self.num_timesteps,
            dropout=self.dropout
        )



        self.head = RegressionHead(
            n_prop=self.n_prop,
            dim = self.out_channels
        )

        self.flat = nn.Flatten()
        

    def forward(self, feats, apply_head = True):
        
        x = self.encoder(
            feats.x,
            feats.edge_index, 
            feats.edge_attr,
            feats.batch
        )

        if apply_head:
            preds = self.head(x)
            
        else:
            preds = x
        return preds