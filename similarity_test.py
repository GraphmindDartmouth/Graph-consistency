from dataset import *
from datetime import datetime
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data,Dataset
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from train import train_splits,validate_splits
from perturbation import PertubateRandom

def load_model(load_path):
    #print("Load model from: ",load_path)
    model= torch.load(load_path,map_location=torch.device('cpu'))
    model.return_embeds=True
    return model

def compute_cosine_similarity(graph_representation):
    # Normalize the graph representations to unit vectors
    graph_representation_norm = F.normalize(graph_representation, p=2, dim=1)
    cosine_sim_matrix = torch.mm(graph_representation_norm, graph_representation_norm.t())
    identity = torch.eye(cosine_sim_matrix.size(0), device=cosine_sim_matrix.device)
    
    # Use the identity matrix to zero out diagonal elements
    cosine_sim_matrix = cosine_sim_matrix * (1 - identity)
    
    return cosine_sim_matrix


def load_dataset(dataset_name,split_num):
    dataset_path = f"./datasplit/{dataset_name}/split{str(split_num)}.pt"
    dataset = torch.load(dataset_path)
    return dataset

def rankloss_consistency(dataset_list,model_list,type="rank"):


    if type=="rank":
        store_dic="./similarity/"
        load_dict="model&dataset"
    else:
        store_dic="./norank_similarity/"
        load_dict="model&dataset"

    for datasetname in dataset_list:

        if os.path.exists(store_dic+datasetname)==False:
            os.mkdir(store_dic+datasetname)
        for model in model_list:

            model_record=[]
            
            # print(model,type)

            similarity_list=[]
            for i in range(5):
                dataset=load_dataset(datasetname ,i)

                model_rankloss=load_model(f"./{load_dict}/{model}/{datasetname}_nodist_{str(i)}.pkl")
                model_rankloss.medium=True
                model_rankloss.eval()
                model_rankloss.medium=1
                model_rankloss.return_embeds=False
                model_rankloss.to("cpu")

                test_data=dataset[int(len(dataset)*(train_splits)):int(len(dataset)*(train_splits+validate_splits))]

                per_result_rank=[[] for i in range(2)]
                pred_rank=[]
                with torch.no_grad():
                    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
                    for data in tqdm(test_loader):
                        if type=='rank':
                            data,medium=model_rankloss(data)
                        else:
                            data,medium=model_rankloss(data)
                        pred = data.argmax(dim=1) 
                        pred_rank.append(data.item() for data in pred)
                        for i in range(2):
                            per_result_rank[i].append(medium[i])

                feature_matrix_rank=[(torch.cat(per_result_rank[i])).squeeze(1)  for i in range(2)]

                cosine_rank=[compute_cosine_similarity(feature_matrix_rank[i]) for i in range(2)]

                model_record.append(cosine_rank)
            torch.save(model_record,store_dic+datasetname+"/"+model+".pt")
            print(f"Similarity on model {model} /dataset {datasetname} saved, type: {type}")
            print(np.mean(similarity_list))



    

if __name__=="__main__":
    dataset_list=["DD"]
    model_list=["GMT","GraphSAGE"]

    # rankloss_consistency(dataset_list,model_list,type="rank")
    rankloss_consistency(dataset_list,model_list,type="norank")
