#only experiment on graph classification tasks
import torch
from model import *
from torch_geometric.loader import DataLoader

from tqdm import tqdm
import torch.nn as nn
from utils import EarlyStopper,get_histogram,get_node_number
import wandb
import time

TUData=["PROTEINS","IMDB-BINARY","IMDB-MULTI","COLLAB","NCI1","NCI109","COIL-RAG", "DD"]
OGB_Data=["ogbg-molhiv"]
SNAP_Data = ["reddit_threads"]

#proportion of training set
train_splits=0.8  
validate_splits=0.1
criterion = torch.nn.CrossEntropyLoss()
save_path="./model/"

cls_criterion = torch.nn.BCEWithLogitsLoss
reg_criterion = torch.nn.MSELoss
multicls_criterion = torch.nn.CrossEntropyLoss


def get_model(model_name,dataset,device,config):
    dropout_ratio=config['dropout']
    hidden_size=config['hidden_size']
    num_layers=config['num_layers']
    num_classes = dataset.num_classes if dataset.name not in ['ogbg-ppa','twitch_egos'] else len(torch.unique(dataset.y))
    num_features = dataset.num_features if dataset.name not in ['ogbg-ppa','twitch_egos'] else 1
    edge_attr_dim =  dataset.edge_attr.size()[1] if dataset.name == 'ogbg-ppa' else None
    
    if model_name == 'GCN':
        model = GCN(num_features,hidden_size,num_classes,num_layers=num_layers,dropout=dropout_ratio)
    elif model_name == 'GraphSAGE':
        model = GraphSAGE(num_features,hidden_size,num_classes,num_layers=num_layers,dropout=dropout_ratio)
    elif model_name == 'GIN':
        model = GIN(num_features,hidden_size,num_classes,num_layers=num_layers,dropout=dropout_ratio)
    elif model_name == 'GTransformer':
        model = TransformerNet(num_features,hidden_size,num_classes,num_layers=num_layers,dropout=dropout_ratio)
    elif model_name=='GMT':
        if (dataset.name in TUData) or (dataset.name in SNAP_Data):
            model = GraphMultisetTransformer(num_features,hidden_size,num_classes,config['heads'],avg_num_nodes=np.ceil([np.mean([data.num_nodes for data in dataset])]))
        elif dataset.name in OGB_Data:
            model = GraphMultisetTransformer_for_OGB(num_features,hidden_size,num_classes,num_heads=config['heads'],avg_num_nodes=np.ceil([np.mean([data.num_nodes for data in dataset])]), edge_attr_dim=edge_attr_dim)
        else:
            raise ValueError("This Model is not implemented")    
    else:
        raise ValueError("This Model is not implemented")
    
    print(model_name, " Training...")
    
    return model.to(device)

def train(model, data_loader, optimizer, device, task_type=None):
    model=model.to(device)
    model.train()

    print('Training...')
        
    for step, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        data = data.to(device)
        if data.x.shape[0] == 1 or data.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            y = data.y
            loss = 0
            out = model(data)
            
            
            if task_type is None:
                loss = F.cross_entropy(out, y)
            elif task_type == 'binary classification':
                y = y.to(torch.float32).squeeze()
                is_labeled = y==y
                loss = cls_criterion()(out.squeeze()[is_labeled], y[is_labeled])
            elif task_type == 'multiclass classification':
                y = y.to(torch.int64).squeeze()
                pred = out.squeeze()
                is_labeled = y==y
                loss = multicls_criterion()(pred[is_labeled], y[is_labeled])    
            else:
                raise 
            
            #loss=nn.CrossEntropyLoss(out[is_labeled], batch.y[is_labeled])
            loss.backward()
            optimizer.step()
        
        
      

def test(model,loader,device,evaluator=None, task_type=None):
    model.eval()

    correct = 0
    total_loss=0
    total_samples = 0
    print("Testing...")
    with torch.no_grad():
        pred_list = []
        y_list = []
        for data in tqdm(loader, total=len(loader)): 
            data=data.to(device)
            y = data.y
            test_loss = 0
            out = model(data)
            
            if task_type is None:
                test_loss = F.cross_entropy(out, y)
            elif task_type == 'binary classification':
                y = y.to(torch.float32).squeeze()
                is_labeled = y == y
                test_loss = cls_criterion()(out.squeeze()[is_labeled], y[is_labeled])
            elif task_type == 'multiclass classification':
                y = y.to(torch.int64).squeeze()
                pred = out.squeeze()
                is_labeled = y==y
                test_loss = multicls_criterion()(pred[is_labeled], y[is_labeled])                
            else:
                raise 
                
            total_loss += test_loss.item()  
            


            if evaluator is not None:
                pred_list.append(out)
            else:
                pred_list.append(out.argmax(dim=1))
            y_list.append(data.y)
             
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
        return correct / y_list.size(0), total_loss / len(loader)
    else:
        return correct, total_loss / len(loader)


def train_model(model_name,dataset,config,dataloaders,patience=30,
                min_delta=0.005,device=7,wandb_record=True,save_model=False,
                seed=1234, fromstore=False,evaluator=None, task_type=None, repeat_time=0
                ):
    
    epoch = config['epochs'] 
    
    
    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")


    print("Model: ",model_name, "Dataset: ",dataset.name )
    print("Train on device:",device)

    if save_model:

        #save_path='./cka_model/'+model_name+'/'
        save_path='./model&dataset/'+model_name+'/'
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)
        
        if wandb_record:
            saved_folder_path=save_path+dataset.name+'/'
            os.mkdir(saved_folder_path) if os.path.exists(saved_folder_path)==False else None
            model_saved_path=saved_folder_path+str(wandb.run.name)+".pkl"
        else:
            model_saved_path=save_path+dataset.name+f"_nodist_{repeat_time}.pkl"
    else:
        model_saved_path=None
    
    
    [train_loader, valid_loader, test_loader] = dataloaders

    model = get_model(model_name,dataset,device=device,config=config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'],betas=(0.9, 0.999))
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta,file_path=model_saved_path,saved=save_model)
    

    for epoch in range(1, epoch):
        print(f'{model_name}, {dataset.name}')
        train(model,train_loader,optimizer,device,task_type)
        train_acc, train_loss = test(model,train_loader,device,evaluator,task_type)
        test_acc, _ = test(model,test_loader,device,evaluator,task_type)
        val_acc, validation_loss= test(model,valid_loader,device,evaluator,task_type)

        if wandb_record:
            dict_to_log={"train_acc": train_acc,"train_loss": train_loss, "valid_loss": validation_loss,"test_acc": test_acc,"val_acc": val_acc}  
            wandb.log(dict_to_log)

        if early_stopper.early_stop(val_acc, epoch, test_acc, model):
            break

        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss:{train_loss:.4f}, Test Acc: {test_acc:.4f}', "Validation Loss: ",validation_loss, "acc_early_stop: ",early_stopper.test_acc_record)
    
    print("Early stopping at Epoch: %d, Test Acc: %f"%(early_stopper.epoch_counter, early_stopper.test_acc_record))

    if wandb_record:  
        wandb.log({"test_acc_of_early_stop": early_stopper.test_acc_record})
    return early_stopper.test_acc_record
    


def data_loader(model_name,train_dataset,test_dataset,valid_dataset,batch_size):
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_loader= DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, valid_loader

