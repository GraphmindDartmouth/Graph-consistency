import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MLP, GINConv, global_add_pool, GINConv,SAGEConv,TransformerConv
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential
import numpy as np 
from utils import *
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#----------------------------------------------------------------
#if we want to observe the output of GNN directly, we can should not add non-linear activation function before our 
# observation. 
#----------------------------------------------------------------

#For reference, see https://github.com/LingxiaoShawn/GraphClassificationBenchmark
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False,medium=False):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.hidden_channels = hidden_dim
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(
                GCNConv(hidden_dim, hidden_dim))   
        
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.return_embeds = return_embeds
        self.medium=medium
        self.dist_matrix=[]


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, edge_weight=None ):
        x, adj_t,batch=data.x.to(dtype=torch.float), data.edge_index, data.batch

        self.dist_matrix=[]
        for conv in self.convs[:]:
            x = conv(x, adj_t, edge_weight=edge_weight)
            x = F.relu(x)

            x_embed=global_add_pool(x, batch)
            if self.medium:
                self.dist_matrix.append(x_embed)

        x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)    
        x = self.lin2(x)
       
        # out = x if self.return_embeds else F.log_softmax(self.lin2(x), dim=-1)
        if self.medium:
            return x,self.dist_matrix
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False,medium=False):
        super(GraphSAGE, self).__init__()


        self.num_layers = num_layers
        self.hidden_channels = hidden_dim
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(
                SAGEConv(hidden_dim, hidden_dim))   
             

        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)

        self.dropout = dropout
        self.dist_matrix=[]
        self.return_embeds = return_embeds
        self.medium=medium

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, adj_t,batch=data.x.to(dtype=torch.float), data.edge_index, data.batch
        self.dist_matrix=[]
        for conv in self.convs[:]:
            x = conv(x, adj_t,)
            x = F.relu(x)
            x_embed=global_add_pool(x, batch)
            if self.medium:
                self.dist_matrix.append(x_embed)

        x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)    
        x = self.lin2(x)
       
        # out = x if self.return_embeds else F.log_softmax(self.lin2(x), dim=-1)
        if self.medium:
            return x,self.dist_matrix
        return x

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,dropout,
                return_embeds=False,medium=False):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.num_layers=num_layers
        mlp = Sequential(
                Linear(input_dim, hidden_dim),
                ReLU(),
                BatchNorm( hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
            )
        self.convs.append(GINConv(mlp, train_eps=True))
        
        for i in range(self.num_layers-1):
            mlp = Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                BatchNorm( hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
            )
            conv = GINConv(mlp, train_eps=True)
            self.convs.append(conv)

        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)
        self.drop_out=dropout
        self.return_embeds=return_embeds
        self.dist_matrix=[]
        self.medium=medium
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, edge_weight=None ):
        x, adj_t,batch=data.x.to(dtype=torch.float), data.edge_index, data.batch
        self.dist_matrix=[]

        for conv in self.convs[:]:
            x = conv(x, adj_t,)
            x = F.relu(x)
            x_embed=global_add_pool(x, batch)
            # distance_matrix=compute_cosine_similarity(x_embed)
            # self.dist_matrix.append(distance_matrix)
            if self.medium:
                self.dist_matrix.append(x_embed)
            

        x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.drop_out, training=self.training)    
        x = self.lin2(x)
        
        # out = x if self.return_embeds else F.log_softmax(self.lin2(x), dim=-1)
        if self.medium:
            return x,self.dist_matrix
        return x
    
class TransformerNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False, medium=False):
        super(TransformerNet, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_dim
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            TransformerConv(input_dim, hidden_dim, heads=1, concat=True, dropout=dropout)
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                TransformerConv(hidden_dim * 1, hidden_dim, heads=1, concat=True, dropout=dropout)
            )
        
        self.lin1 = Linear(hidden_dim * 1, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.return_embeds = return_embeds
        self.medium = medium

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, edge_attr=None):
        x, edge_index, batch = data.x.to(dtype=torch.float), data.edge_index, data.batch
        self.dist_matrix = []
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x_embed=global_add_pool(x, batch)
            if self.medium:
                self.dist_matrix.append(x_embed)

        x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        # out = x if self.return_embeds else F.log_softmax(self.lin2(x), dim=-1)
        if self.medium:
            return x,self.dist_matrix
        return x
    
