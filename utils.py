from torch.nn import Linear
import numpy as np
import torch
import os
import random
from torch_geometric.utils import degree
import torch_geometric.transforms as T
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import degree
from utils import *
from grakel import Graph
import torch.nn as nn
import pandas as pd 

class EarlyStopper:
    def __init__(self, patience, min_delta,file_path,saved=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = -np.inf
        self.epoch_counter =0
        self.test_acc_record = 0
        self.file_path = file_path
        self.model_saved=saved

        self.model_updated=False

    def early_stop(self, validation_loss,epoch_num, test_acc_record,model):
        print("max_validation_acc: %f, validation_acc: %f"%(self.min_validation_loss, validation_loss))
        if validation_loss > self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.epoch_counter=epoch_num
            self.counter = 0
            self.test_acc_record = test_acc_record

            if self.model_saved==True:
                torch.save(model,self.file_path)
                self.model_updated=True

        elif validation_loss < (self.min_validation_loss + self.min_delta):
            self.counter += 1
            
        if self.counter >= self.patience:
            return True
            
        return False

'''
Data processing for TUdataset follows https://github.com/JinheonBaek/GMT/blob/main/utils/data.py,
assign one-hot code for data without feature
''' 
class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def record_result(model_name, dataset_name, res, root):
    if not os.path.exists(root):
        with open(root, 'w') as f:
            res_df = pd.DataFrame(data={dataset_name: res}, index=[model_name])
            res_df.to_csv(root)
            return 
    
    res_df = pd.read_csv(root, index_col=0)
    part_data = pd.DataFrame(data={dataset_name: res}, index=[model_name])
    con1 = model_name in res_df.index
    con2 = dataset_name in res_df.columns
    print(con1, con2)
    if con1 and con2:
        res_df[dataset_name][model_name] = res
    if con1 and not con2:
        res_df = pd.concat([res_df, part_data], axis=1)
    if not con1 and con2:
        res_df = pd.concat([res_df, part_data], axis=0)
    if not con1 and not con2:
        res_df = pd.concat([res_df, part_data])
        
    res_df.to_csv(root)
    
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_seed(seed):
    """Sets seed"""
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def visualize_graph(G, edge_list,file_name):
    """
    Visualize a NetworkX graph with specific edges highlighted.

    Parameters:
    G (networkx.Graph): A NetworkX graph.
    edge_list (list of tuples): A list of edges to highlight.
    """
    # Draw the graph with default edge color
    if G.is_directed():
        G=G.to_undirected()
    pos = nx.spring_layout(G)  # You can choose other layouts as per your preference
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')

    # Highlight the specified edges
    nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color='red', width=2)

    plt.savefig(file_name)
    plt.clf()


def distance_compute(tensor1, tensor2):
    pdist = torch.nn.PairwiseDistance(p=2)
    return pdist(tensor1, tensor2)

def pyg_to_grakel_optimized(pyg_data_list):
    """Convert a list of PyG Data objects directly to GraKeL graphs."""
    G_grakel = []  # List to store GraKeL graphs

    for data in pyg_data_list:
        edge_set = {(int(e[0]), int(e[1])) for e in data.edge_index.t().tolist()}
        node_labels = {i: str(data.x[i].tolist()) if data.x is not None else '0' for i in range(data.num_nodes)}
        edge_labels = {(u, v): '1' for u, v in edge_set}
        G_grakel.append(Graph(edge_set, node_labels=node_labels, edge_labels=edge_labels))
    return G_grakel

def pyg_to_grakel(pyg_data_list):
    grakel_graphs = []
    for data in pyg_data_list:
        # Get the adjacency matrix from the edge index
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        adj_matrix[edge_index[0], edge_index[1]] = 1
        
        # Get node attributes (if any)
        node_attributes = data.x.numpy() if 'x' in data else None
        
        # Create a GraKeL graph
        graph = Graph(adj_matrix.numpy(), node_labels=node_attributes, )
        grakel_graphs.append(graph)
    
    return grakel_graphs


class RankNetLoss(nn.Module):
    """
    RankNet loss implemented as a PyTorch nn.Module for integration into PyTorch models.
    """
    def __init__(self):
        super(RankNetLoss, self).__init__()

    def forward(self, previous_layer, current_layer):
        """
        Compute the RankNet loss.
        
        Parameters:
        - labels: Tensor of labels indicating the preferred item (1 if first item is preferred, 0 otherwise).
        - similarities: Tensor of similarities for each item pair.
        
        Returns:
        - RankNet loss for the input batch.
        """
        # Calculate the pairwise differences between all elements in similarities tensor
        pairwise_diffs = current_layer.unsqueeze(1) - current_layer.unsqueeze(0)
        
        # Apply the sigmoid function to these differences
        sigmoid_diffs = torch.sigmoid(pairwise_diffs)
        
        # Generate the label matrix where element (i, j) is 1 if labels[i] > labels[j], and -1 otherwise
        label_matrix = previous_layer.unsqueeze(1) - previous_layer.unsqueeze(0)
        label_matrix = label_matrix.sign()
        
        # Compute the RankNet loss
        losses = -label_matrix * torch.log(sigmoid_diffs + 1e-15) - (1 - label_matrix) * torch.log(1 - sigmoid_diffs + 1e-15)
        
        # Filter out the NaNs caused by log(0) and compute the mean loss
        losses = torch.where(torch.isnan(losses), torch.zeros_like(losses), losses)
        mean_loss = torch.mean(losses)
        
        return mean_loss