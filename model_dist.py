import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MLP, GINConv, PNAConv,global_add_pool, GINConv,GATConv,SAGEConv,TransformerConv,GraphMultisetTransformer,EGConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch.nn import BatchNorm1d as BatchNorm
from torch_geometric.nn import BatchNorm as BatchNorm2
from torch.nn import Linear, ReLU, Sequential
from torch.nn import LayerNorm
from torch_scatter import scatter
from torch_geometric.utils import to_dense_adj,to_dense_batch
import numpy as np 
from utils import *
from torch_geometric.utils import dense_to_sparse
import os
import torch
from torch.nn import Linear
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.utils import get_laplacian
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from utils_dist import *

#For reference, see https://github.com/LingxiaoShawn/GraphClassificationBenchmark
# def compute_cosine_similarity(graph_representation,device="cpu") :
#     # Aggregate node features to get a single representation for each graph
   
#     num_graphs = graph_representation.shape[0]
#     cosine_sim_matrix = torch.zeros((num_graphs, num_graphs), device=graph_representation.device)

#     # Compute pairwise cosine similarity
#     for i in range(num_graphs):
#         cosine_sim_matrix[i] = F.cosine_similarity(graph_representation[i], graph_representation)
#     mask= torch.ones((cosine_sim_matrix.shape[0],cosine_sim_matrix.shape[0]),device=cosine_sim_matrix.device)-torch.eye(cosine_sim_matrix.shape[0],device=cosine_sim_matrix.device).to(cosine_sim_matrix.device)
#     cosine_sim_matrix=cosine_sim_matrix*mask
    
    
#     return cosine_sim_matrix


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False, reg_term=0,loss_module=None):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_dim
        self.alpha = reg_term
        self.convs = torch.nn.ModuleList([GCNConv(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.return_embeds = return_embeds
        self.loss_module=loss_module

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, edge_weight=None):
        x, edge_index, batch = data.x.to(dtype=torch.float), data.edge_index, data.batch
        layer_outputs = []  # To store output of each layer

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            layer_outputs.append(x)

        # Apply global pooling to get graph-level embeddings
        pooled_outputs = [global_add_pool(layer_output, batch) for layer_output in layer_outputs]
        
        pooled_output = F.relu(self.lin1(pooled_outputs[-1]))  # Only use the last layer's output for final prediction
        out = self.lin2(F.dropout(pooled_output, p=self.dropout, training=self.training))

        # return F.log_softmax(out, dim=-1), pooled_outputs
        return out, pooled_outputs

    def loss(self, pred, y, pooled_outputs, task_type=None):
        return loss(pred, y, pooled_outputs, task_type, self.loss_module, self.alpha)

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False, reg_term=0.01,loss_module=None):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.return_embeds = return_embeds
        self.dropout = dropout
        self.reg_term = reg_term

        # Initial convolution layer
        mlp = Sequential(Linear(input_dim, hidden_dim), ReLU(), BatchNorm(hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.convs.append(GINConv(mlp, train_eps=True))
        
        # Remaining convolution layers
        for _ in range(num_layers - 1):
            mlp = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), BatchNorm(hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp, train_eps=True))
        
        # Linear layers for final prediction
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)
        self.loss_module=loss_module

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, edge_weight=None):
        x, edge_index, batch = data.x.to(dtype=torch.float), data.edge_index, data.batch
        layer_outputs = []  # To store output of each layer

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            layer_outputs.append(x)

        # Apply global pooling to get graph-level embeddings
        pooled_outputs = [global_add_pool(layer_output, batch) for layer_output in layer_outputs]
        pooled_output = F.relu(self.lin1(pooled_outputs[-1]))  # Only use the last layer's output for final prediction
        out = self.lin2(F.dropout(pooled_output, p=self.dropout, training=self.training))

        return out, pooled_outputs

    def loss(self, pred, y, pooled_outputs, task_type=None):
        return loss(pred, y, pooled_outputs, task_type, self.loss_module, self.reg_term)

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embed=False, reg_term=0,loss_module=None):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.return_embeds = return_embed
        self.alpha = reg_term

        # Define the convolutional layers
        self.convs = torch.nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Define the linear layers
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)
        self.loss_module=loss_module

    def forward(self, data, edge_weight=None):
        x, edge_index, batch = data.x.to(dtype=torch.float), data.edge_index, data.batch
        layer_outputs = []  # To store output of each layer

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            layer_outputs.append(x)

        # Apply global pooling to get graph-level embeddings
        pooled_outputs = [global_add_pool(layer_output, batch) for layer_output in layer_outputs]
        pooled_output = F.relu(self.lin1(pooled_outputs[-1]))  # Only use the last layer's output for final prediction
        out = self.lin2(F.dropout(pooled_output, p=self.dropout, training=self.training))

        return out, pooled_outputs

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def loss(self, pred, y, pooled_outputs, task_type=None):
        return loss(pred, y, pooled_outputs, task_type, self.loss_module, self.alpha)

class TransformerNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False, reg_term=0, loss_module=None):
        super(TransformerNet, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_dim
        self.alpha = reg_term  # Regularization term
        self.convs = torch.nn.ModuleList([
            TransformerConv(input_dim, hidden_dim, heads=1, concat=True, dropout=dropout)
        ])
        for _ in range(num_layers - 1):
            self.convs.append(
                TransformerConv(hidden_dim, hidden_dim, heads=1, concat=True, dropout=dropout)
            )
        
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.return_embeds = return_embeds
        self.loss_module = loss_module  # Custom loss module for regularization

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, edge_attr=None):
        x, edge_index, batch = data.x.to(dtype=torch.float), data.edge_index, data.batch
        layer_outputs = []  # To store output of each layer

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            layer_outputs.append(x)

        # Apply global pooling to get graph-level embeddings
        pooled_outputs = [global_add_pool(layer_output, batch) for layer_output in layer_outputs]
        
        # Final prediction layer
        pooled_output = F.relu(self.lin1(pooled_outputs[-1]))
        x = F.dropout(pooled_output, p=self.dropout, training=self.training)
        out = self.lin2(x)
        
        # out = out if self.return_embeds else F.log_softmax(out, dim=-1)
        return out, pooled_outputs  # Return both output and all layer outputs


    def loss(self, pred, y, pooled_outputs, task_type=None):
        return loss(pred, y, pooled_outputs, task_type, self.loss_module, self.alpha)


    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, dropout, return_embeds=False, reg_term=0,loss_module=None):  

        super(MEWIS, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.append(GINConv(MLP_MEWIS(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True)))
        self.alpha = reg_term
        self.loss_module = loss_module
        
        for _ in range(num_layers):
            self.pools.append(MEWISPool(hidden_dim=hidden_dim))
            self.convs.append(GINConv(MLP_MEWIS(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True)))
            
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        main_loss = 0
        layer_outputs = []
        for i in range(len(self.convs)-1):
            gc = self.convs[i]
            pool = self.pools[i]
            x = gc(x, edge_index)
            x = torch.relu(x)
            x_pooled, edge_index_pooled, batch_pooled, loss, mewis = pool(x, edge_index, batch)
            layer_outputs.append((x_pooled, batch_pooled))
            
            x = x_pooled
            edge_index = edge_index_pooled
            batch = batch_pooled 
            main_loss = main_loss + loss
            
        x = self.convs[-1](x, edge_index)
        x = torch.relu(x)
        layer_outputs.append((x,batch_pooled))

        readout = torch.cat([x[batch == i].mean(0).unsqueeze(0) for i in torch.unique(batch)], dim=0)

        out = self.fc1(readout)
        out = torch.relu(out)
        out = self.fc2(out)
        
        pooled_outputs = [global_add_pool(layer_output, layer_batch) for (layer_output,layer_batch) in layer_outputs]

        return out, pooled_outputs, loss

    def loss(self, pred, y, pooled_outputs, task_type=None):
        return loss(pred, y, pooled_outputs, task_type, self.loss_module, self.alpha)