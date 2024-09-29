import torch
import torch_geometric
from torch_geometric.datasets import TUDataset
import torch.nn.functional as F
from tqdm import tqdm
import os
from utils import *
from torch_geometric.data import InMemoryDataset, Data
from ogb.graphproppred import GraphPropPredDataset
from dataset_ogb import PygGraphPropPredDataset
from torch_geometric import transforms as T
from torch_geometric.utils import degree, to_dense_adj
from utils import *
import pandas as pd 
import json

TUData=["PROTEINS","IMDB-BINARY","IMDB-MULTI","REDDIT-BINARY","COLLAB","COIL-RAG", "COIL-DEL",
        "NCI1","NCI109","MUTAG","DD","PTC_MR","REDDIT-MULTI-5K",'ENZYME','Letter-high']
OGB_Data=['ogbg-molhiv','ogbg-molpcba','ogbg-ppa','ogbg-code2']
SNAP_Data = ['twitch_egos', 'reddit_threads']
##zinc=[]


def file_initialize():
    if os.path.exists("./tmp"):
        return    
    else:
        os.mkdir("./tmp")

def load_dataset(dataset_name,model_name,shuffle=False,):

    file_initialize()
    #print(file_name)
    if dataset_name in TUData:
        dataset = get_TUdataset(dataset_name,model_name,shuffle=shuffle)
    elif dataset_name in OGB_Data:
        dataset = get_OGBdataset(dataset_name, model_name, shuffle=shuffle)
    elif dataset_name in SNAP_Data:
        dataset = SNAP_dataset(dataset_name,transform=add_degree)
    else:
        print("Error")
        raise Exception("Error in load_dataset")
    
    return dataset

def add_zeros(data):
    data.x = torch.zeros((data.num_nodes,1), dtype=torch.long)
    return data

def add_degree(data):
    x = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long) + degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    data.x = x.reshape((data.num_nodes,1))
    return data

def get_OGBdataset(name, model_name, sparse=True, cleaned=False, normalize=False,Permutation=None,index=None, shuffle=True, pre_transform=None):
    
    if model_name=="GPS":
        transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        dataset = PygGraphPropPredDataset(name, root=os.path.join('./tmp'), pre_transform=transform, skip_collate=False)
    else:
        # dataset = PygGraphPropPredDataset(name, root=os.path.join('./tmp'), pre_transform=pre_transform, skip_collate=False)
        dataset = PygGraphPropPredDataset(name, root=os.path.join('./tmp'), pre_transform=pre_transform, skip_collate=False, transform=add_degree if name=='ogbg-ppa' else None)
        
    if normalize:
        dataset.data.x -= torch.mean(dataset.data.x, axis=0)
        dataset.data.x /= torch.std(dataset.data.x, axis=0)
    
    # if name=='ogbg-ppa':
    #     dataset.data.x = torch.zeros((dataset.data.num_nodes,1))
    
    return dataset
    

def get_TUdataset(name, model_name, sparse=True, cleaned=False, normalize=False,Permutation=None,index=None,shuffle=True,pre_transform=None):

    
    if model_name=="GPS":
        transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        dataset = TUDataset(os.path.join('./tmp/pe'), name, use_node_attr=True, pre_transform=transform,cleaned=cleaned)
    else:
        dataset = TUDataset(os.path.join('./tmp'), name, use_node_attr=True, pre_transform=pre_transform,cleaned=cleaned)

    dataset.data.edge_attr = None

    if shuffle:
        dataset = ShuffleDataset(dataset)
        
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    elif normalize:

        dataset.data.x -= torch.mean(dataset.data.x, axis=0)
        dataset.data.x /= torch.std(dataset.data.x, axis=0)

    if not sparse:
        max_num_nodes = 0
        for data in dataset:
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        if dataset.transform is None:
            dataset.transform = T.ToDense(max_num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(max_num_nodes)])
    
    if name=='NCI1' or name=='NCI109': 
        dataset=dataset.shuffle()
    return dataset


class ShuffleDataset(InMemoryDataset):

    def __init__(self, tu_dataset):
        super(ShuffleDataset, self).__init__('.', None, None)
        self.tu_dataset = tu_dataset
        self.name=tu_dataset.name
        self.data, self.slices = self.process_dataset()

    def process_dataset(self):
        indices = torch.randperm(len(self.tu_dataset))
        data_list = [self.tu_dataset[i] for i in indices]
        return self.collate(data_list)

class SNAP_dataset(InMemoryDataset):
    def __init__(self, dataset_name='twitch_egos', root='tmp', transform=None):
        super(SNAP_dataset, self).__init__(root, transform)
        self.name = dataset_name
        self.data = None
        
        with open(os.path.join(root, dataset_name, f"{dataset_name.split('_')[0]}_edges.json")) as f:
            self.data = json.load(f)
            print('load successfully!')
        self.y = torch.tensor(pd.read_csv(os.path.join(root, dataset_name, f"{dataset_name.split('_')[0]}_target.csv"))['target']).unsqueeze(0).T
        self.transform = transform

    def len(self):
        return len(self.data)

    def get(self, idx):

        edge_index = torch.tensor(self.data[str(idx)]).T
        label = self.y[idx]
        data = Data(edge_index=edge_index, y=label, num_nodes=torch.max(edge_index)+1)

        if self.transform:
            data = self.transform(data)

        return data


def load_splited_dataset(folder):
    dataset = torch.load(os.path.join(folder, 'data.pt'))
    return dataset

def load_datasets(root_dir='data_splits'):
    train_dataset = load_dataset(os.path.join(root_dir, 'train'))
    val_dataset = load_dataset(os.path.join(root_dir, 'val'))
    test_dataset = load_dataset(os.path.join(root_dir, 'test'))
    return train_dataset, val_dataset, test_dataset

