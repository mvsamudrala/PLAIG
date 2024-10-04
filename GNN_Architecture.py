import os
import re
import pandas as pd
import random
import subprocess
from rdkit import Chem
from rdkit.Chem import Descriptors
import networkx as nx
import torch
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from torch_geometric.data import Data, Dataset
import torch_geometric.utils as pyg_utils
from torch.nn import Linear, L1Loss, MSELoss
from torch_geometric.nn import GeneralConv, GATv2Conv, NNConv, GINEConv, global_mean_pool
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Pad
# from Bio import PDB
from biopandas.pdb import PandasPdb
from prody import parsePDBHeader
from typing import Optional
import numpy as np
# from mordred import Calculator, descriptors
import binana
from openbabel import openbabel
import time
import warnings
import GNN_representationv3
import GNN_representationv4
from io import StringIO
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score
import pickle
import openpyxl
import joblib


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout_rate, num_node_features, num_edge_features,
                 num_ligand_features, num_pocket_features):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GeneralConv(num_node_features, hidden_channels, in_edge_channels=num_edge_features)
        self.convs = torch.nn.ModuleList(
            [GeneralConv(hidden_channels, hidden_channels, in_edge_channels=num_edge_features) for _ in
             range(num_layers - 1)])
        self.global_fc_ligand = Linear(num_ligand_features, hidden_channels)
        self.global_fc_pocket = Linear(num_pocket_features, hidden_channels)
        self.fusion_fc = Linear(3 * hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, edge_attr, batch, ligand_features, pocket_features, return_embeddings):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_attr=edge_attr).relu()
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr).relu()
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        global_ligand_features_transformed = self.global_fc_ligand(ligand_features)
        global_pocket_features_transformed = self.global_fc_pocket(pocket_features)
        x = torch.cat([x, global_ligand_features_transformed, global_pocket_features_transformed], dim=1)
        x = self.fusion_fc(x)
        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if not return_embeddings:
            x = self.lin(x)
        # x = self.lin(x)

        return x