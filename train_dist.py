#only experiment on graph classification tasks
import torch
from model import GCN, GIN,GraphSAGE,TransformerNet
from GMT_model.nets import GraphMultisetTransformer, GraphMultisetTransformer_for_OGB
from torch_geometric.loader import DataLoader
import os
from tqdm import tqdm
import torch.nn as nn
from utils import EarlyStopper
import wandb
import torch.nn.functional as F
import numpy as np
from utils_dist import *
import torch._dynamo
torch._dynamo.config.suppress_errors = True

TUData=["PROTEINS","IMDB-BINARY","IMDB-MULTI","COLLAB","NCI1","NCI109","COIL-RAG", "DD"]
OGB_Data=["ogbg-molhiv"]
SNAP_Data = ["reddit_threads"]

    

class RankNetLoss(nn.Module):
    """
    RankNet loss implemented as a PyTorch nn.Module, excluding self-comparisons by removing diagonal elements.
    """
    def __init__(self):
        super(RankNetLoss, self).__init__()

    @torch.compile
    def forward(self, previous_layer, current_layer):
        """
        Parameters:
            - previous_layer: Tensor of labels indicating the preferred item (1 if first item is preferred, 0 otherwise).
            - current_layer: Tensor of item features or similarities.
        
        Returns:
            - RankNet loss for the input batch, excluding self-comparisons.
        """
        n = current_layer.size(0)
        
        mask = ~torch.eye(n, dtype=torch.bool, device=current_layer.device)

        pairwise_diffs = current_layer.unsqueeze(1) - current_layer.unsqueeze(0)
        pairwise_diffs = pairwise_diffs * mask.unsqueeze(2).float()

        #pairwise_diff
        sigmoid_diffs = torch.sigmoid(pairwise_diffs)
        label_matrix = previous_layer.unsqueeze(1) - previous_layer.unsqueeze(0)
        label_matrix = label_matrix.sign()
        label_matrix = (1 + label_matrix) / 2
        label_matrix = label_matrix * mask.float()

        losses = -label_matrix * torch.log(sigmoid_diffs + 1e-15) - (1 - label_matrix) * torch.log(1 - sigmoid_diffs + 1e-15)
        losses = torch.where(torch.isnan(losses) | ~mask, torch.zeros_like(losses), losses)
        mean_loss = losses.sum() / (mask.sum()*n)
        return mean_loss
    


#proportion of training set
train_splits=0.8  
validate_splits=0.1
criterion = torch.nn.CrossEntropyLoss()
save_path="./model/"

    
def get_model(model_name, dataset, device, config, loss_module="RankNetLoss"):
    dropout_ratio=config['dropout']
    hidden_size=config['hidden_size'] 
    num_layers=config['num_layers']
    reg_term=config['reg_term']
    num_classes = dataset.num_classes if dataset.name != 'ogbg-ppa' else len(torch.unique(dataset.y))
    
    if loss_module == "RankNetLoss":
        loss_func=RankNetLoss().to(device)
        print("Using RankNetLoss")
 
    if model_name == 'GCN':
        model = GCN(dataset.num_features,hidden_size,num_classes,num_layers=num_layers,dropout=dropout_ratio,medium=True)
    elif model_name == 'GraphSAGE':
        model = GraphSAGE(dataset.num_features,hidden_size,num_classes,num_layers=num_layers,dropout=dropout_ratio,medium=True)
    elif model_name == 'GIN':
        model = GIN(dataset.num_features,hidden_size,num_classes,num_layers=num_layers,dropout=dropout_ratio,medium=True)
    elif model_name == 'GTransformer':
        model = TransformerNet(dataset.num_features,hidden_size,num_classes,num_layers=num_layers,dropout=dropout_ratio,medium=True)
    elif model_name=='GMT':
        if (dataset.name in TUData) or (dataset.name in SNAP_Data):
            model = GraphMultisetTransformer(dataset.num_features,hidden_size,num_classes,config['heads'],avg_num_nodes=np.ceil([np.mean([data.num_nodes for data in dataset])]),medium=True)
        elif dataset.name in OGB_Data:
            model = GraphMultisetTransformer_for_OGB(dataset.num_features,hidden_size,num_classes,num_heads=config['heads'],avg_num_nodes=np.ceil([np.mean([data.num_nodes for data in dataset])]), edge_attr_dim=7, medium=True)
        else:
            raise ValueError("This Model is not implemented")    
    else:
        raise ValueError("This Model is not implemented")

    model_decorator(model, reg_term, loss_func)
    print(model_name, " Training...")
    return model

def train(model, data_loader, optimizer, device, task_type):
    model = model.to(device)
    model.train()
    print('Training...')
    for step, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        data = data.to(device)
        if data.x.shape[0] == 1 or data.batch[-1] == 0:
            continue  # Skip batches that are too small or have incorrect batching
        
        optimizer.zero_grad()
        loss = 0
        out, pooled_outputs = model(data)


        sub_loss ,_ = model.loss(out, data.y, pooled_outputs=pooled_outputs, task_type=task_type)
        loss = loss + sub_loss

        loss.backward()

        optimizer.step()


def test(model, loader, device, evaluator=None, task_type=None):
    model.eval()

    correct = 0
    total_loss=0
    total_samples = 0
    total_dist_loss=0

    with torch.no_grad():
        pred_list = []
        y_list = []
        print("Testing...")

        for data in tqdm(loader, total=len(loader)):
            data = data.to(device)

            out, pooled_outputs = model(data)

            test_loss, dist_loss = model.loss(out, data.y, pooled_outputs, task_type)
            total_dist_loss += dist_loss.item()
            total_loss += test_loss.item()
            
            if evaluator is not None:
                # for ogb data
                pred_list.append(out)
            else:
                pred_list.append(out.argmax(dim=1))
            y_list.append(data.y)
    
            # correct += int((pred == data.y).sum())
            total_samples += data.y.size(0)  # Update total samples for accuracy calculation
        
        pred_list = torch.cat(pred_list)
        y_list = torch.cat(y_list)
        if evaluator is not None:
            if y_list.size() != pred_list.size():
                pred_list = torch.argmax(pred_list, dim=1).unsqueeze(0).T
            test_loss = evaluator.eval({"y_true": y_list, "y_pred": pred_list})
            correct = test_loss[list(test_loss.keys())[0]]
        else:
            correct = torch.sum(pred_list==y_list).item()

    if evaluator is None:
        return correct / total_samples, total_loss / len(loader), total_dist_loss / len(loader)
    else:
        return correct, total_loss / len(loader), total_dist_loss / len(loader)


def train_model_dist(model_name,dataset,dataloaders,config,patience=100,
                min_delta=0.005,device=7,wandb_record=True,save_model=False,
                seed=12345, fromstore=False, evaluator=None,task_type=None, repeat_time=0
                ):
    
    
    epoch = config['epochs']
    
    
    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
    print("Model: ",model_name, "Dataset: ",dataset.name )
    print("Train on device:",device)

    if save_model:

        save_path='./model&dataset/'+model_name+'/'
        #save_path='./cka_model/'+model_name+'/'
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)
        
        if wandb_record:
            saved_folder_path=save_path+dataset.name+'/'
            os.mkdir(saved_folder_path) if os.path.exists(saved_folder_path)==False else None
            model_saved_path=saved_folder_path+str(wandb.run.name)+".pkl"
        else:
            model_saved_path=save_path+dataset.name+f"_distrecord_{repeat_time}.pkl"
    else:
        model_saved_path=None
    
    [train_loader, valid_loader, test_loader] = dataloaders
    # model = get_model(model_name,dataset,dropout,hidden_size,num_layers=num_layers,device=device,reg_term=loss_reg_term)
    model = get_model(model_name,dataset,device=device,config=config)

    # model.reset_parameters()


    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'],betas=(0.9, 0.999))
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta,file_path=model_saved_path,saved=save_model)

    for epoch in range(0, epoch):
        print(f'{model_name}, {dataset.name}')
        train(model,train_loader,optimizer,device,task_type)
        
        # train_acc, train_loss, train_dist_loss = test(model,train_loader,device,evaluator,task_type)
        train_acc=0
        train_loss=0
        train_dist_loss=0
        test_acc, _ ,test_dist_loss= test(model,test_loader,device,evaluator,task_type)
        val_acc, validation_loss, valid_dist_loss= test(model,valid_loader,device,evaluator,task_type)
        if not validation_loss==validation_loss:
            break
        if wandb_record:
            dict_to_log={"train_acc": train_acc,"train_loss": train_loss, "valid_loss": validation_loss,"test_acc": test_acc,"val_acc": val_acc,
                         "train_dist_loss":train_dist_loss,"test_dist_loss":test_dist_loss,}  
            wandb.log(dict_to_log)
        # if epoch % 10 == 0:
        #     save_path="./dist_perserve/"+str(epoch)+".pt"
        #    torch.save(model,save_path)
        if early_stopper.early_stop(val_acc,epoch, test_acc, model):
            break

        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss:{train_loss:.4f}, Test Acc: {test_acc:.4f}', "Test_dist_loss: ",test_dist_loss,"Validation Loss: ",validation_loss,"acc_early_stop: ",early_stopper.test_acc_record)
    print("Early stopping at Epoch: %d, Test Acc: %f"%(early_stopper.epoch_counter, early_stopper.test_acc_record))
    
    with open(f"record/{model_name}_{config['reg_term']}.txt", 'a+') as file:
        file.write(f"{early_stopper.test_acc_record}\n")
    
    
    #torch.save(model,model_saved_path)

    if wandb_record:  
        wandb.log({"test_acc_of_early_stop": early_stopper.test_acc_record})
    
    return early_stopper.test_acc_record

def data_loader(dataset_name, dataset, batch_size):
    
    if (dataset_name in TUData) or (dataset_name in SNAP_Data):
        #proportion of training set
        train_splits=0.8  
        validate_splits=0.1
        data_train_index=int(len(dataset)*train_splits)
        data_test_index=int(len(dataset)*(train_splits+validate_splits))
        
        train_dataset =dataset[:data_train_index]
        test_dataset = dataset[data_train_index:data_test_index]
        valid_dataset = dataset[data_test_index:]
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        valid_loader= DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        return [train_loader, valid_loader, test_loader]
    
    elif dataset_name in OGB_Data:
        split_idx = dataset.get_idx_split()

        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)

        return [train_loader, valid_loader, test_loader]
    
    else:
        print("Error")
        raise Exception("Error in load_dataset")

