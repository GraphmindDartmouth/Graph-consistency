import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp

from torch_geometric.utils import to_dense_batch

from GMT_model.layers import SAB, ISAB, PMA
from GMT_model.layers import GCNConv_for_OGB, GINConv_for_OGB

from ogb.graphproppred.mol_encoder import AtomEncoder

from math import ceil
from utils_dist import *

class GraphRepresentation(torch.nn.Module):

    def __init__(self, num_features, num_hidden, num_classes, ):

        super(GraphRepresentation, self).__init__()

        self.num_features = num_features
        self.nhid = num_hidden
        self.num_classes = num_classes
        self.pooling_ratio = 0.25
        self.dropout_ratio = 0.5

    def get_convs(self):

        convs = nn.ModuleList()

        _input_dim = self.num_features
        _output_dim = self.nhid

        for _ in range(2):
            
            conv = GCNConv(_input_dim, _output_dim)

            convs.append(conv)

            _input_dim = _output_dim
            _output_dim = _output_dim

        return convs

    def get_pools(self):

        pools = nn.ModuleList([gap])

        return pools

    def get_classifier(self):

        return nn.Sequential(
            nn.Linear(self.nhid, self.nhid),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid//2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid//2, self.num_classes)
        )

class GraphMultisetTransformer(GraphRepresentation):

    def __init__(self, num_features, num_hidden, num_classes, num_heads,avg_num_nodes,reg_term,loss_module):

        super(GraphMultisetTransformer, self).__init__(num_features, num_hidden, num_classes, )

        self.ln = False
        self.num_heads = num_heads
        self.cluster = True
        self.alpha=reg_term
        self.loss_module=loss_module

        self.model_sequence = 'GMPool_G-SelfAtt-GMPool_I'.split('-') 
        self.avg_num_nodes = avg_num_nodes
        self.convs = self.get_convs()
        self.pools = self.get_pools()
        self.classifier = self.get_classifier()

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # For Graph Convolution Network
        xs = []

        for _ in range(2):

            x = F.relu(self.convs[_](x.to(torch.float), edge_index))
            xs.append(x)

        # For jumping knowledge scheme
        pooled_outputs = [global_add_pool(layer_output, batch) for layer_output in xs]
        x = torch.cat(xs, dim=1)

        # For Graph Multiset Transformer
        for _index, _model_str in enumerate(self.model_sequence):

            if _index == 0:

                batch_x, mask = to_dense_batch(x, batch)

                extended_attention_mask = mask.unsqueeze(1)
                extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

            if _model_str == 'GMPool_G':

                batch_x = self.pools[_index](batch_x, attention_mask=extended_attention_mask, graph=(x, edge_index, batch))

            else:

                batch_x = self.pools[_index](batch_x, attention_mask=extended_attention_mask)

            extended_attention_mask = None

        batch_x = self.pools[len(self.model_sequence)](batch_x)
        x = batch_x.squeeze(1)

        # For Classification
        x = self.classifier(x)

        return x, pooled_outputs
    
    def loss(self, pred, y, pooled_outputs, task_type=None):
            return loss(pred, y, pooled_outputs, task_type, self.loss_module, self.alpha)
    
    def get_pools(self, _input_dim=None, reconstruction=False):

        pools = nn.ModuleList()

        _input_dim = self.nhid * 2 if _input_dim is None else _input_dim # TODO: *3
        _output_dim = self.nhid
        _num_nodes = ceil(self.pooling_ratio * self.avg_num_nodes)

        for _index, _model_str in enumerate(self.model_sequence):

            if (_index == len(self.model_sequence) - 1) and (reconstruction == False):
                
                _num_nodes = 1

            if _model_str == 'GMPool_G':

                pools.append(
                    PMA(_input_dim, self.num_heads, _num_nodes, ln=self.ln, cluster=self.cluster, mab_conv='GCN')
                )

                _num_nodes = ceil(self.pooling_ratio * _num_nodes)

            elif _model_str == 'GMPool_I':

                pools.append(
                    PMA(_input_dim, self.num_heads, _num_nodes, ln=self.ln, cluster=self.cluster, mab_conv=None)
                )

                _num_nodes = ceil(self.pooling_ratio * _num_nodes)

            elif _model_str == 'SelfAtt':

                pools.append(
                    SAB(_input_dim, _output_dim, self.num_heads, ln=self.ln, cluster=self.cluster)
                )

                _input_dim = _output_dim
                _output_dim = _output_dim

            else:

                raise ValueError("Model Name in Model String <{}> is Unknown".format(_model_str))

        pools.append(nn.Linear(_input_dim, self.nhid))

        return pools

class GraphMultisetTransformer_for_OGB(GraphMultisetTransformer):

    def __init__(self, num_features, num_hidden, num_classes, num_heads,avg_num_nodes,reg_term,loss_module,edge_attr_dim):

        super(GraphMultisetTransformer_for_OGB, self).__init__(num_features, num_hidden, num_classes, num_heads,avg_num_nodes,reg_term,loss_module)

        self.atom_encoder = AtomEncoder(self.nhid)
        self.alpha=reg_term
        self.loss_module=loss_module
        self.edge_attr_dim = edge_attr_dim

        self.convs = self.get_convs()

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.atom_encoder(x)

        # For Graph Convolution Network
        xs = []

        for _ in range(2):

            x = F.relu(self.convs[_](x, edge_index, edge_attr))
            xs.append(x)

        # For jumping knowledge scheme
        x = torch.cat(xs, dim=1)
        pooled_outputs = [global_add_pool(layer_output, batch) for layer_output in xs]
        # For Graph Multiset Transformer
        for _index, _model_str in enumerate(self.model_sequence):

            if _index == 0:

                batch_x, mask = to_dense_batch(x, batch)

                extended_attention_mask = mask.unsqueeze(1)
                extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

            if _model_str == 'GMPool_G':

                batch_x = self.pools[_index](batch_x, attention_mask=extended_attention_mask, graph=(x, edge_index, batch))

            else:

                batch_x = self.pools[_index](batch_x, attention_mask=extended_attention_mask)

            extended_attention_mask = None

        batch_x = self.pools[len(self.model_sequence)](batch_x)
        x = batch_x.squeeze(1)

        # For Classification
        x = self.classifier(x)

        return x, pooled_outputs

    def loss(self, pred, y, pooled_outputs, task_type=None):
        return loss(pred, y, pooled_outputs, task_type, self.loss_module, self.alpha)
    
    def get_convs(self):

        convs = nn.ModuleList()

        for _ in range(2):
            convs.append(GCNConv_for_OGB(self.nhid, edge_attr_dim=self.edge_attr_dim))

        return convs
