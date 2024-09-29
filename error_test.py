from dataset import *
from datetime import datetime
from torch_geometric.loader import DataLoader,DenseDataLoader
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data,Dataset
import torch
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from train import train_splits,validate_splits
from perturbation import PertubateRandom
from utils import distance_compute
from scipy.stats import spearmanr,kendalltau
from utils import * 
import json
from grakel.kernels import WeisfeilerLehman,PropagationAttr,WeisfeilerLehmanOptimalAssignment



dataset_list=["NCI109",]#"MUTAG", "PTC_MR"] #"NCI109","NCI1","NCI109","NCI1",
model_list=["GIN","GCN","GAT","GraphSage","MixHop"]# "DenseGraphSage","DenseGIN","DenseGraphConv","GCN2"

#device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
saved_model_path="./model/"
log_file_path="./logs/"

def load_model(model_name, dataset_name,index):
    file_name="run"+str(index)
    load_path=saved_model_path+model_name+'/'+dataset_name+'/'+file_name+".pkl"

    #print("Load model from: ",load_path)
    model= torch.load(load_path,map_location=torch.device('cpu'))
    model.return_embeds=True
    return model

type_a="pr"

results = {}

for dataset_name in dataset_list:
    for model_name in model_list:

        print("Model: ",model_name," Dataset: ",dataset_name)
        #model=load_model(model,dataset)
        load_path=os.path.join("tmp",dataset_name,model_name)
        dataset_all=torch.load(os.path.join(load_path,"splited.pt"))
        data_train_index=int(len(dataset_all)*train_splits)
        data_test_index=int(len(dataset_all)*(train_splits+validate_splits))
        test_dataset=dataset_all[data_train_index:data_test_index]     


        kernel1 = WeisfeilerLehmanOptimalAssignment(n_iter=5,normalize=True)
        grakel_list=pyg_to_grakel_optimized(test_dataset)
        K1 = kernel1.fit_transform(grakel_list)

        model_error_simialrity=[]
        model_correct_simialrity=[]

        for model_id in range(0,10):
            single_count=0
            model=load_model(model_name,dataset_name,model_id)
            model.to(device)
            model.eval()
            model.return_embeds=False
            
            error_pred_list=[]

            wrong_pred_similarities = []
            correct_pred_similarities = []

            for idx, data in tqdm(enumerate(test_dataset)):
                label = data.y
                pred = model(data.x, data.edge_index, data.batch)
                pred_label = pred.argmax(dim=1) # Assuming pred is a tensor of logits or probabilities

                # Find indices of data with the same true label
                same_label_indices = [i for i, d in enumerate(test_dataset) if d.y.item() == label and i != idx]
                
                if not torch.equal(pred_label, label):
                    # Incorrect prediction
                    if same_label_indices:
                        # Calculate mean similarity to true data with the same label (excluding itself)
                        mean_similarity = np.mean([K1[idx, i] for i in same_label_indices])
                        wrong_pred_similarities.append(mean_similarity)
                else:
                    # Correct prediction
                    if same_label_indices:
                        # Calculate mean similarity to true data with the same label (excluding itself)
                        mean_similarity = np.mean([K1[idx, i] for i in same_label_indices])
                        correct_pred_similarities.append(mean_similarity)

            # Calculate the overall mean similarities
                
            mean_similarity_wrong = np.mean(wrong_pred_similarities) if wrong_pred_similarities else 0
            mean_similarity_correct = np.mean(correct_pred_similarities) if correct_pred_similarities else 0

            model_correct_simialrity.append(mean_similarity_correct)
            model_error_simialrity.append(mean_similarity_wrong)

        print(f"Mean similarity for wrong predictions: {np.mean(model_error_simialrity)}")
        print(f"Mean similarity for correct predictions: {np.mean(model_correct_simialrity)}")


