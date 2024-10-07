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
import GNN_representationv5
from GNN_Architecture import GNN
from io import StringIO
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score
import pickle
import openpyxl
import joblib
import seaborn as sns


def normalize_graph_features(ligand_features, pocket_features, normalization_stats):
    ligand_features_means = normalization_stats["Ligand Means"]
    pocket_features_means = normalization_stats["Pocket Means"]
    ligand_features_sds = normalization_stats["Ligand Sds"]
    pocket_features_sds = normalization_stats["Pocket Sds"]
    ligand_features_array = np.array(ligand_features)
    pocket_features_array = np.array(pocket_features)

    # numeric_mask = np.array([np.issubdtype(ligand_features_array[:, i].dtype, np.number)
    #                          for i in range(ligand_features_array.shape[1])])
    # ligand_features_array = ligand_features_array[:, numeric_mask]
    #
    # numeric_mask = np.array([np.issubdtype(pocket_features_array[:, i].dtype, np.number)
    #                          for i in range(pocket_features_array.shape[1])])
    # pocket_features_array = pocket_features_array[:, numeric_mask]

    # ligand_features_means = np.mean(ligand_features_array, axis=0)
    # pocket_features_means = np.mean(pocket_features_array, axis=0)
    # print(ligand_features_means)
    # print(pocket_features_means)
    # ligand_features_sds = np.std(ligand_features_array, axis=0)
    # ligand_features_sds = np.where(ligand_features_sds == 0, 1, ligand_features_sds)
    # pocket_features_sds = np.std(pocket_features_array, axis=0)
    # pocket_features_sds = np.where(pocket_features_sds == 0, 1, pocket_features_sds)
    # print(ligand_features_sds)
    # print(pocket_features_sds)
    ligand_features_standardized = (ligand_features_array - ligand_features_means) / ligand_features_sds
    pocket_features_standardized = (pocket_features_array - pocket_features_means) / pocket_features_sds

    return ligand_features_standardized, pocket_features_standardized


def pca_graph_features(ligand_features, pocket_features):
    ligand_df = pd.DataFrame(ligand_features)
    pocket_df = pd.DataFrame(pocket_features)
    pca = PCA(n_components=22)
    pca_ligand_output = pca.fit_transform(ligand_df)
    # explained_ligand_variance = pca.explained_variance_ratio_
    # cumulative_ligand_variance = np.cumsum(explained_ligand_variance)
    # print(explained_ligand_variance)
    # plt.plot(cumulative_ligand_variance)
    # plt.xlabel('Number of Components')
    # plt.ylabel('Cumulative Explained Variance')
    # plt.show()

    pca = PCA(n_components=13)
    pca_pocket_output = pca.fit_transform(pocket_df)
    # explained_pocket_variance = pca.explained_variance_ratio_
    # cumulative_pocket_variance = np.cumsum(explained_pocket_variance)
    # print(explained_pocket_variance)
    # plt.plot(cumulative_pocket_variance)
    # plt.xlabel('Number of Components')
    # plt.ylabel('Cumulative Explained Variance')
    # plt.show()

    return pca_ligand_output, pca_pocket_output


def prepare_data(complex_graph, pdb_code):
    data = pyg_utils.from_networkx(complex_graph)
    # data.x = torch.tensor([complex_graph.nodes[n]['x'] for n in complex_graph.nodes], dtype=torch.float)
    # data.edge_attr = torch.tensor([complex_graph.edges[ed]['edge_attr'] for ed in complex_graph.edges], dtype=torch.float)
    # graph_attr = [complex_graph.graph[pair] for pair in complex_graph.graph]

    # data.graph_attr = torch.tensor(graph_attr, dtype=torch.float)
    data.ligand_attr = torch.tensor(complex_graph.graph["ligand_attr"], dtype=torch.float).unsqueeze(0)
    data.pocket_attr = torch.tensor(complex_graph.graph["pocket_attr"], dtype=torch.float).unsqueeze(0)
    # binding_affinity = index_dataframe.loc[index_dataframe['PDB Code'] == pdb_code, '-logKd/Ki'].iloc[0]
    # print(binding_affinity)
    # data.y = torch.tensor([binding_affinity], dtype=torch.float)
    data.name = pdb_code
    # print(data.batch)

    return data


# directory = "/Users/mvsamudrala/BindingAffinityGNN/refined-set"
# entries = os.listdir(directory)
# subdirectories_refined = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
# subdirectories_refined.remove("index")
# subdirectories_refined.remove("readme")
# print(len(subdirectories_refined))
#
# # subdirectories_refined = subdirectories_refined[0:200]
#
# index_file = "/Users/mvsamudrala/BindingAffinityGNN/refined-set/index/INDEX_refined_data.2020"
# with open(index_file, 'r') as file:
#     text = file.read()
#     text = text[text.find("2r58"):]
#     index_df = pd.read_csv(StringIO(text), sep='\s+', header=None, names=["PDB Code", "resolution", "release year", "-logKd/Ki", "Kd/Ki", "slash", "reference", "ligand name"], index_col=False)
#     print(index_df)
#
# directory_general = "/Users/mvsamudrala/BindingAffinityGNN/v2020-general-PL"
# entries_general = os.listdir(directory_general)
# subdirectories_general = [entry for entry in entries_general if os.path.isdir(os.path.join(directory_general, entry))]
# subdirectories_general.remove("index")
# subdirectories_general.remove("readme")
#
# index_file_general = "/Users/mvsamudrala/BindingAffinityGNN/pdb_key_general"
# with open(index_file_general, 'r') as file:
#     text = file.read()
#     text = text[text.find("3zzf"):]
#     index_df_general = pd.read_csv(StringIO(text), sep='\s+', header=None, names=["PDB Code", "resolution", "release year", "-logKd/Ki", "Kd/Ki", "slash", "reference", "ligand name"], index_col=False)
#     index_df_general = index_df_general[index_df_general["Kd/Ki"].str.contains("Kd|Ki")].reset_index(drop=True)
#     print(index_df_general)
#
# subdirectories_general = [code for code in subdirectories_general if code in index_df_general["PDB Code"].values]
# print(len(subdirectories_general))
# # subdirectories_general = subdirectories_general[0:200]
# subdirectories = subdirectories_refined + subdirectories_general
#
# index_df = pd.concat([index_df, index_df_general], ignore_index=True)

# directory_path = "/Users/mvsamudrala/BindingAffinityGNN/DUDE_Z_PDB"
# receptors = [name for name in os.listdir(directory_path)]
# del receptors[4]
# # receptors = receptors[0:5]
# # print(receptors)
# receptor_directories = [os.path.join(directory_path, name) for name in receptors]
# # print(receptor_directories)
# #
# dudez_dataset = {}
# count = 0
# for receptor_directory in receptor_directories:
#     receptor_name = receptors[count]
#     protein_file = f"{receptor_directory}/{receptor_name}.pdb"
#     # print(protein_file)
#     dudez_dataset[protein_file] = None
#     ligand_decoy_files = []
#     # print(receptor_name)
#     for file in os.listdir(receptor_directory):
#         if file == "ligands":
#             ligand_directory = os.path.join(receptor_directory, file)
#             for ligand in os.listdir(ligand_directory):
#                 ligand_file = os.path.join(ligand_directory, ligand)
#                 # print(ligand_file)
#                 ligand_decoy_files.append(ligand_file)
#         elif file == "decoys":
#             decoy_directory = os.path.join(receptor_directory, file)
#             for decoy in os.listdir(decoy_directory):
#                 decoy_file = os.path.join(decoy_directory, decoy)
#                 # print(decoy_file)
#                 ligand_decoy_files.append(decoy_file)
#         elif "xtal" in file:
#             ligand_file = os.path.join(receptor_directory, file)
#             # print(ligand_file)
#             ligand_decoy_files.append(ligand_file)
#     dudez_dataset[protein_file] = ligand_decoy_files
#     count += 1
#
# for key, value in dudez_dataset.items():
#     print(key)


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
# all_ligand_features_key = {}
# all_pocket_features_key = {}
# dataset = []
no_nodes_count = 0
for protein, ligands in dudez_dataset.items():
    random.shuffle(ligands)
    count = 0
    protein_name = os.path.splitext(os.path.basename(protein))[0]
    print(protein_name)
    protein_pocket_path = protein
    print(protein_pocket_path)
    protein_pocket_pdbqt_path = f"/Users/mvsamudrala/BindingAffinityGNN/PDBQT_DUDE_Z_only/{protein_name}/{protein_name}.pdbqt"
    print(protein_pocket_pdbqt_path)
    # protein_path = "/Users/mvsamudrala/BindingAffinityGNN/refined-set/1a1e/1a1e_protein.pdb"
    # pdbqt_output_dir = "/Users/mvsamudrala/BindingAffinityGNN/PDBQT_refined"
    # path_parts = protein_pocket_path.split("/")
    # pdb_code = path_parts[-2]
    for ligand in ligands:
        print(count)
        ligand_name = os.path.splitext(os.path.basename(ligand))[0]
        print(ligand_name)
        ligand_path = ligand
        print(ligand_path)
        if "xtal" in ligand_name:
            ligand_pdbqt_path = f"/Users/mvsamudrala/BindingAffinityGNN/PDBQT_DUDE_Z_only/{protein_name}/{ligand_name}.pdbqt"
        elif "ZINC" in ligand_name:
            # ligand_pdbqt_path = f"/Users/mvsamudrala/BindingAffinityGNN/PDBQT_DUDE_Z_only/{protein_name}/decoys/{ligand_name}.pdbqt"
            continue
        else:
            if protein_name == "AMPC" or protein_name == "DRD4" or protein_name == "MT1":
                ligand_pdbqt_path = f"/Users/mvsamudrala/BindingAffinityGNN/PDBQT_DUDE_Z_only/{protein_name}/ligands/{ligand_name}.pdbqt"
            else:
                continue
        print(ligand_pdbqt_path)
        print()
        try:
            graph = GNN_representationv5.pl_complex_to_graph(protein_pocket_path, ligand_path, protein_pocket_pdbqt_path, ligand_pdbqt_path, distance_cutoff)
            # for edge in graph.edges(data=True):
            #     print(edge)
            if len(graph.nodes) == 0:
                no_nodes_count += 1
                print(f"Graph has no nodes, #{no_nodes_count}")
                continue
            pre_dataset.append((graph, (protein_name, ligand_name)))
            # all_ligand_features_key[code] = graph.graph["ligand_attr"]
            # all_pocket_features_key[code] = graph.graph["pocket_attr"]
            print(graph.graph["ligand_attr"])
            all_ligand_features.append(graph.graph["ligand_attr"])
            print(graph.graph["pocket_attr"])
            all_pocket_features.append(graph.graph["pocket_attr"])
            # in_channels = graph_data.num_node_features
            # edge_channels = graph_data.edge_features.shape[1]
            # hidden_channels = 32
            # out_channels = 1
            # num_graph_features = len(graph_data.graph_features)
            #
            # model = GCN(in_channels, edge_channels, hidden_channels, out_channels, num_graph_features)
            #
            # print(model)

        except Exception as e:
            cannot_read_mols.append((protein_name, ligand_name))
            print(f"Cannot read {protein_name}, {ligand_name} file: {e}")
        # if count == 1000:
        #     break
        count += 1
print()
all_ligand_features_normalized, all_pocket_features_normalized = normalize_graph_features(all_ligand_features, all_pocket_features, normalization_statistics)
for index, (graph, (protein_name, ligand_name)) in enumerate(pre_dataset):
    graph.graph["ligand_attr"] = all_ligand_features_normalized[index]
    print(graph.graph["ligand_attr"])
    graph.graph["pocket_attr"] = all_pocket_features_normalized[index]
    print(graph.graph["pocket_attr"])
    graph_data = prepare_data(graph, f"{protein_name}, {ligand_name}")
    # count = 0
    # for edge in graph.edges():
    #     print(count)
    #     print(edge)
    #     count += 1
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
graph_data_file = "DUDE_Z_norm_3_pt2.pkl"
with open(graph_data_file, 'wb') as file:
    pickle.dump(dataset, file)

# start = time.time()
# warnings.filterwarnings('ignore')
# cannot_read_mols = []
# count = 0
# distance_cutoff = 3
# pre_dataset = []
# all_ligand_features = []
# all_pocket_features = []
# # all_ligand_features_key = {}
# # all_pocket_features_key = {}
# dataset = []
# print(len(subdirectories))
# for code in subdirectories:
#     if code in subdirectories_refined:
#         print(count)
#         print(code)
#         protein_pocket_path = f"/Users/mvsamudrala/BindingAffinityGNN/refined-set-hydrogenated/{code}_hydrogenated_pocket.pdb"
#         protein_pocket_pdbqt_path = f"/Users/mvsamudrala/BindingAffinityGNN/PDBQT_refined_hydrogens/{code}_pocket.pdbqt"
#         protein_path = "/Users/mvsamudrala/BindingAffinityGNN/refined-set/1a1e/1a1e_protein.pdb"
#         pdbqt_output_dir = "/Users/mvsamudrala/BindingAffinityGNN/PDBQT_refined"
#         # path_parts = protein_pocket_path.split("/")
#         # pdb_code = path_parts[-2]
#         ligand_path = f"/Users/mvsamudrala/BindingAffinityGNN/refined-set-hydrogenated/{code}_hydrogenated_ligand.pdb"
#         ligand_pdbqt_path = f"/Users/mvsamudrala/BindingAffinityGNN/PDBQT_refined_hydrogens/{code}_ligand.pdbqt"
#     else:
#         print(count)
#         print(code)
#         protein_pocket_path = f"/Users/mvsamudrala/BindingAffinityGNN/general-set-hydrogenated/{code}_hydrogenated_pocket.pdb"
#         protein_pocket_pdbqt_path = f"/Users/mvsamudrala/BindingAffinityGNN/PDBQT_general/{code}_pocket.pdbqt"
#         # protein_path = "/Users/mvsamudrala/BindingAffinityGNN/refined-set/1a1e/1a1e_protein.pdb"
#         # pdbqt_output_dir = "/Users/mvsamudrala/BindingAffinityGNN/PDBQT_general"
#         # path_parts = protein_pocket_path.split("/")
#         # pdb_code = path_parts[-2]
#         ligand_path = f"/Users/mvsamudrala/BindingAffinityGNN/general-set-hydrogenated/{code}_hydrogenated_ligand.pdb"
#         ligand_pdbqt_path = f"/Users/mvsamudrala/BindingAffinityGNN/PDBQT_general/{code}_ligand.pdbqt"
#     try:
#         graph = GNN_representationv5.pl_complex_to_graph(protein_pocket_path, ligand_path, protein_pocket_pdbqt_path, ligand_pdbqt_path, distance_cutoff)
#         # for edge in graph.edges(data=True):
#         #     print(edge)
#         pre_dataset.append((graph, code))
#         # all_ligand_features_key[code] = graph.graph["ligand_attr"]
#         # all_pocket_features_key[code] = graph.graph["pocket_attr"]
#         print(graph.graph["ligand_attr"])
#         all_ligand_features.append(graph.graph["ligand_attr"])
#         print(graph.graph["pocket_attr"])
#         all_pocket_features.append(graph.graph["pocket_attr"])
#         # in_channels = graph_data.num_node_features
#         # edge_channels = graph_data.edge_features.shape[1]
#         # hidden_channels = 32
#         # out_channels = 1
#         # num_graph_features = len(graph_data.graph_features)
#         #
#         # model = GCN(in_channels, edge_channels, hidden_channels, out_channels, num_graph_features)
#         #
#         # print(model)
#
#     except Exception as e:
#         cannot_read_mols.append(code)
#         print(f"Cannot read {code} file: {e}")
#     # if count == 1000:
#     #     break
#     count += 1
# print()
# all_ligand_features_normalized, all_pocket_features_normalized, ligand_means, pocket_means, ligand_sds, pocket_sds = normalize_graph_features(all_ligand_features, all_pocket_features)
# for index, (graph, code) in enumerate(pre_dataset):
#     graph.graph["ligand_attr"] = all_ligand_features_normalized[index]
#     print(graph.graph["ligand_attr"])
#     graph.graph["pocket_attr"] = all_pocket_features_normalized[index]
#     print(graph.graph["pocket_attr"])
#     graph_data = prepare_data(graph, index_df, code)
#     # count = 0
#     # for edge in graph.edges():
#     #     print(count)
#     #     print(edge)
#     #     count += 1
#     print(f"Data object: {graph_data}")
#     print(f'Number of nodes: {graph_data.num_nodes}')
#     print(f'Number of edges: {graph_data.num_edges}')
#     print(f'Average node degree: {graph_data.num_edges / graph_data.num_nodes:.2f}')
#     print(f'Has isolated nodes: {graph_data.has_isolated_nodes()}')
#     print(f'Has self-loops: {graph_data.has_self_loops()}')
#     print(f'Is undirected: {graph_data.is_undirected()}')
#     print()
#     dataset.append(graph_data)
# print(ligand_means)
# print(pocket_means)
# print(ligand_sds)
# print(pocket_sds)
# normalization_statistics = {"Ligand Means": ligand_means, "Pocket Means": pocket_means, "Ligand Sds": ligand_sds, "Pocket Sds": pocket_sds}
# normalization_statistics_file = "combined_set_normalization_statistics.pkl"
# with open(normalization_statistics_file, 'wb') as file:
#     pickle.dump(normalization_statistics, file)
#
# with open(normalization_statistics_file, 'rb') as file:
#     dataset = pickle.load(file)
# for key, value in dataset.items():
#     print(key)
#     print(value)


graph_data_file = "DUDE_Z_norm_3_pt2.pkl"
with open(graph_data_file, 'rb') as file:
    dataset = pickle.load(file)
print(len(dataset))
random.shuffle(dataset)

num_hidden_channels = 256
batch_size = 32
num_layers = 4
optimizer = "Adagrad"
lr = 0.001
loss_func = "L1Loss"
num_epochs = 50
dropout_rate = 0.2

test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
model = GNN(hidden_channels=256, num_layers=4, dropout_rate=0.2,
            num_node_features=40, num_edge_features=11, num_ligand_features=88, num_pocket_features=74)

model.load_state_dict(torch.load("GNN_Model7.pth"))
print(model)
model.eval()
embeddings = []
labels = []
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
# labels = np.hstack(labels)
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
# print(f"Refined Set Testing PCC: {stack_pcc}")

general_set_output = {"Receptor": protein_labels, "Ligand": ligand_labels, "Predicted BA": stack_predictions}
df_general_set = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in general_set_output.items()]))

df_non_zinc = df_general_set[~df_general_set['Ligand'].str.contains("ZINC", case=False)]
df_zinc = df_general_set[df_general_set['Ligand'].str.contains('ZINC', case=False)]
df_non_zinc['Group'] = 'Active Ligand'
df_zinc['Group'] = 'Decoy'

df_combined = pd.concat([df_non_zinc, df_zinc])
t_stat, p_value = ttest_ind(df_zinc['Predicted BA'], df_non_zinc['Predicted BA'], nan_policy='omit')

plt.figure(figsize=(8, 6))
sns.violinplot(x='Group', y='Predicted BA', data=df_combined)

plt.title("DUDE-Z")
plt.xlabel('Ligand Group')
plt.ylabel('Predicted Binding Affinity')
if p_value < 0.05:
    plt.text(0.5, max(df_combined['Predicted BA']), f'p-value < 0.05',
             ha='center', va='bottom', fontsize=12, color='black')
else:
    plt.text(0.5, max(df_combined['Predicted BA']), f'p-value = {p_value:.3f}',
             ha='center', va='bottom', fontsize=12, color='black')

plt.savefig(f'/Users/mvsamudrala/BindingAffinityGNN/Violin_Plots/violin_plot_DUDE_Z_t_test.jpg', format='jpg',
            dpi=600)

plt.show()

df_receptors = {receptor: df for receptor, df in df_general_set.groupby("Receptor")}
# print(df_receptors)
with pd.ExcelWriter('DUDE_Z_Testing6.xlsx', engine='openpyxl') as w:
    for key in df_receptors.keys():
        df_receptors[key].to_excel(w, sheet_name=f"{key}", index=False)
for receptor, df in df_receptors.items():
    df_non_zinc = df[~df['Ligand'].str.contains("ZINC", case=False)]
    df_zinc = df[df['Ligand'].str.contains('ZINC', case=False)]

    df_non_zinc['Group'] = 'Active Ligand'
    df_zinc['Group'] = 'Decoy'

    df_combined = pd.concat([df_non_zinc, df_zinc])
    t_stat, p_value = ttest_ind(df_zinc['Predicted BA'], df_non_zinc['Predicted BA'], nan_policy='omit')

    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Group', y='Predicted BA', data=df_combined)

    plt.title(f'{receptor}')
    plt.xlabel('Ligand Group')
    plt.ylabel('Predicted Binding Affinity')
    if p_value < 0.05:
        plt.text(0.5, max(df_combined['Predicted BA']), f'p-value < 0.05',
                 ha='center', va='bottom', fontsize=12, color='black')
    else:
        plt.text(0.5, max(df_combined['Predicted BA']), f'p-value = {p_value:.3f}',
                 ha='center', va='bottom', fontsize=12, color='black')

    plt.savefig(f'/Users/mvsamudrala/BindingAffinityGNN/Violin_Plots/violin_plot_{receptor}_t_test.jpg', format='jpg', dpi=600)

    plt.show()
end = time.time()
print(end - start)












