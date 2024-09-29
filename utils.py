from torch.nn import Linear
import numpy as np
import torch
import os
import random
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import DataLoader, Data
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import degree, to_undirected
from torch_geometric.utils import dropout_adj
from utils import *
from torch_geometric.utils import to_networkx,dropout_adj,dropout_edge
from grakel import Graph
from torch_geometric.data import Data
from grakel import GraphKernel, graph_from_networkx
import torch.nn as nn
from torch_scatter import scatter

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


#tool function for PNA
def get_histogram(dataset):
    max_degree = -1
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

# Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg
#----------------------------------------------------------------

def get_node_number(dataset):
    max_num=-1
    for data in dataset:
       
        max_num = data.num_nodes
    max_num=max(max_num, 200)
    return max_num


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


def dist_compute(origin_data,data1,data2, model,device=None):
    
    origin_embed=model(origin_data.x,origin_data.edge_index,origin_data.batch)
    embed1=model(data1.x,data1.edge_index,data1.batch)
    embed2=model(data2.x,data2.edge_index,data2.batch)
    dist1=distance_compute(origin_embed,embed1)
    dist2=distance_compute(origin_embed,embed2)
    return dist1,dist2

def remove_edge_bernolli(data,dropout_ratio):
    new_edge_index, _ = dropout_adj(data.edge_index, p=dropout_ratio, force_undirected=False)
    data_dropout = data.clone()
    data_dropout.edge_index = new_edge_index
    return data_dropout


def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x

def drop_edge_weighted(data, edge_weights, p: float, threshold: float = 1.):
    # Scale edge weights and apply threshold
    edge_weights = edge_weights / edge_weights.mean() * p # get the mean value of edge weights
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)#
    # For each edge weight, if it is less than the specified threshold, it remains unchanged. If it is greater than or equal to the threshold, 
    #it is set to the threshold value. This prevents any edge weight from exceeding the threshold.

    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
    new_edge_index = data.edge_index[:, sel_mask]
    
   
    new_data = Data(x=data.x, edge_index=new_edge_index, y=data.y)
    # edges with higher weights are more likely to be removed due to 
    return new_data

def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights


def evc_drop_weights(data):
    evc = eigenvector_centrality(data)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())
def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)

def remove_weighted_edge(data,dropout_ratio, type="eigen"):
    if type=="eigen":
        drop_weights=evc_drop_weights(data)
    elif type=="pr":
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200)
    elif type=="degree":
        drop_weights = degree_drop_weights(data.edge_index)
    
    data1= drop_edge_weighted(data, drop_weights, dropout_ratio)

    return data1

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