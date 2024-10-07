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
import numpy as np
import time
import warnings
import graph_representation
from PLAIG_Architecture import GNN
from io import StringIO
from sklearn.linear_model import LinearRegression
import pickle
import joblib


def normalize_graph_features(ligand_features, pocket_features, normalization_stats):
    ligand_features_means = normalization_stats["Ligand Means"]
    pocket_features_means = normalization_stats["Pocket Means"]
    ligand_features_sds = normalization_stats["Ligand Sds"]
    pocket_features_sds = normalization_stats["Pocket Sds"]
    ligand_features_array = np.array(ligand_features)
    pocket_features_array = np.array(pocket_features)

    ligand_features_standardized = (ligand_features_array - ligand_features_means) / ligand_features_sds
    pocket_features_standardized = (pocket_features_array - pocket_features_means) / pocket_features_sds

    return ligand_features_standardized, pocket_features_standardized


def prepare_data(complex_graph, pdb_code):
    data = pyg_utils.from_networkx(complex_graph)
    data.ligand_attr = torch.tensor(complex_graph.graph["ligand_attr"], dtype=torch.float).unsqueeze(0)
    data.pocket_attr = torch.tensor(complex_graph.graph["pocket_attr"], dtype=torch.float).unsqueeze(0)
    data.name = pdb_code

    return data


def run_model(protein_pdbs, protein_pdbqts, ligand_pdbs, ligand_pdbqts):
    file_lengths = [len(protein_pdbs), len(protein_pdbqts), len(ligand_pdbs), len(ligand_pdbqts)]
    if all(length == file_lengths[0] for length in file_lengths):
        start = time.time()
        normalization_statistics_file = "combined_set_normalization_statistics.pkl"
        with open(normalization_statistics_file, 'rb') as file:
            normalization_statistics = pickle.load(file)
        warnings.filterwarnings('ignore')
        cannot_read_mols = []
        count = 0
        distance_cutoff = 3
        pre_dataset = []
        all_ligand_features = []
        all_pocket_features = []
        dataset = []
        no_nodes_count = 0
        for protein_pdb, protein_pdbqt, ligand_pdb, ligand_pdbqt in zip(protein_pdbs, protein_pdbqts, ligand_pdbs, ligand_pdbqts):
            print(count)
            protein_name = os.path.splitext(os.path.basename(protein_pdb))[0]
            print(protein_name)
            ligand_name = os.path.splitext(os.path.basename(ligand_pdb))[0]
            print(ligand_name)
            try:
                graph = GNN_representationv5.pl_complex_to_graph(protein_pdb, ligand_pdb,
                                                                 protein_pdbqt, ligand_pdbqt,
                                                                 distance_cutoff)
                if len(graph.nodes) == 0:
                    no_nodes_count += 1
                    print(f"Graph has no nodes, #{no_nodes_count}")
                    continue
                pre_dataset.append((graph, (protein_name, ligand_name)))
                print(graph.graph["ligand_attr"])
                all_ligand_features.append(graph.graph["ligand_attr"])
                print(graph.graph["pocket_attr"])
                all_pocket_features.append(graph.graph["pocket_attr"])

            except Exception as e:
                cannot_read_mols.append((protein_name, ligand_name))
                print(f"Cannot read {protein_name}, {ligand_name} file: {e}")
            count += 1
        print()
        all_ligand_features_normalized, all_pocket_features_normalized = normalize_graph_features(all_ligand_features,
                                                                                                  all_pocket_features,
                                                                                                  normalization_statistics)
        for index, (graph, (protein_name, ligand_name)) in enumerate(pre_dataset):
            graph.graph["ligand_attr"] = all_ligand_features_normalized[index]
            print(graph.graph["ligand_attr"])
            graph.graph["pocket_attr"] = all_pocket_features_normalized[index]
            print(graph.graph["pocket_attr"])
            graph_data = prepare_data(graph, f"{protein_name}, {ligand_name}")
            print(f"Data object: {graph_data}")
            print(f'Number of nodes: {graph_data.num_nodes}')
            print(f'Number of edges: {graph_data.num_edges}')
            print(f'Average node degree: {graph_data.num_edges / graph_data.num_nodes:.2f}')
            print(f'Has isolated nodes: {graph_data.has_isolated_nodes()}')
            print(f'Has self-loops: {graph_data.has_self_loops()}')
            print(f'Is undirected: {graph_data.is_undirected()}')
            print()
            dataset.append(graph_data)
        print(len(dataset))
        print("Done creating graphs for each protein-ligand complex, moving on to generating embeddings with GNN...")

        num_hidden_channels = 256
        batch_size = 32
        num_layers = 4
        dropout_rate = 0.2

        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model = GNN(hidden_channels=num_hidden_channels, num_layers=num_layers, dropout_rate=dropout_rate,
                    num_node_features=40, num_edge_features=11, num_ligand_features=88, num_pocket_features=74)

        model.load_state_dict(torch.load("GNN_Model7.pth"))
        print(model)
        model.eval()
        embeddings = []
        pdb_codes = []
        count = 0
        with torch.no_grad():
            for d in test_loader:
                print(count)
                out = model(d.x, d.edge_index, d.edge_attr, d.batch, d.ligand_attr, d.pocket_attr, True)
                out_array = out.cpu().detach().numpy()
                if not np.isnan(out_array).any():
                    embeddings.append(out.cpu().detach().numpy())
                    pdb_codes.append(d.name)
                    print(out_array)
                count += 1

        embeddings = np.vstack(embeddings)
        pdb_codes = np.hstack(pdb_codes)
        print(embeddings)

        stack_model = joblib.load("Stacking_Regressor7.joblib")
        stack_predictions = stack_model.predict(embeddings)
        protein_labels = []
        ligand_labels = []
        for i in range(len(stack_predictions)):
            pdb_string = pdb_codes[i]
            pdb_list = pdb_string.split(", ")
            protein_labels.append(pdb_list[0])
            ligand_labels.append(pdb_list[1])
            print(f"Receptor: {pdb_list[0]}, Ligand: {pdb_list[1]}, Predicted: {stack_predictions[i]}")
        end = time.time()
        print(end - start)

    else:
        print("Not all input lists are the same length. Please make sure each protein-ligand complex has the required "
              "files input into the model.")














