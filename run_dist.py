import argparse
import yaml
import wandb
from dataset import load_dataset,TUData,OGB_Data
from train_dist  import train_model_dist, data_loader
from utils import seed_everything
import os
import torch
from ogb.graphproppred import Evaluator

parser = argparse.ArgumentParser(description='  ')

parser.add_argument('--model', default='GCN', type=str,help='train the XX model')
parser.add_argument('--dataset', default='PROTEINS', type=str,help='on which dataset')
parser.add_argument('--gpu', required=False, default=3, type=int, help='Device Number' )
parser.add_argument("--seed", type=int, default=1234, help="random seed (default: 1234)")
parser.add_argument("--reg_term", type=float, default=0.1, help="random seed (default: 1234)")
parser.add_argument("--repeat_time", type=int, default=10, help="random seed (default: 1234)")
parser.add_argument("--wandb_record", action="store_true")
args = parser.parse_args()

seed_everything(args.seed)

config_path="./rank_config/"+args.model+".yaml"
config=yaml.safe_load(open(config_path,'r'))

wandb_config = {
    "epochs": int(config[args.dataset]["epochs"]),
    "hidden_size":int(config[args.dataset]["hidden_size"]),
    "learning_rate":float(config[args.dataset]["learning_rate"]),
    "dropout":float(config[args.dataset]["dropout"]),
    "batch_size":int(config[args.dataset]["batch_size"]),
    "num_layers":int(config[args.dataset]["num_layers"]),
    "weight_decay":float(config[args.dataset]["weight_decay"]),
    "reg_term":float(config[args.dataset]["reg_term"]),
}

for key in config[args.dataset].keys():
    wandb_config[key] = config[args.dataset][key]


#只在train的时候保存数据
save_data=False

if __name__ == '__main__':
    wandb_record = args.wandb_record
    for item in range(args.repeat_time):
        project_name=args.model+'_'+args.dataset+"_reg_repeat"

        dataset=load_dataset(args.dataset,args.model,shuffle=True,)
        dataloaders = data_loader(args.dataset, dataset, int(wandb_config["batch_size"]))
        if save_data:
            folder="./model&dataset/"+args.dataset+"/"+args.model+'/'+'reg'
            if not os.path.exists(folder):
                os.makedirs(folder)
            torch.save(dataset, os.path.join(folder, f'splited_{item}.pt'))
            print("Shuffled Data saved to ",folder)

        if wandb_record==True:
            run = wandb.init(project=project_name,name="run"+str(item),config=wandb_config)

        if args.dataset in TUData:
            model=train_model_dist(args.model,
                            dataloaders=dataloaders,
                            dataset=dataset,
                            config=dict(wandb_config),
                            device=args.gpu,
                            wandb_record=wandb_record,
                            save_model=True,
                            seed=args.seed,
                            fromstore=False,
                            repeat_time=item, 
                            )     
        
        else:
            train_model_dist(args.model, 
                            dataset,
                            dataloaders=dataloaders,
                            config=dict(wandb_config),
                            device=args.gpu,
                            wandb_record=wandb_record,
                            save_model=True,
                            seed=args.seed,
                            fromstore=False,
                            evaluator=Evaluator(args.dataset) if args.dataset in OGB_Data else None,
                            task_type=dataset.task_type if args.dataset in OGB_Data else None)
        wandb.finish()

    # with open(f'record/{args.model}_{args.reg_term}.txt', 'a+') as file:
    #     file.write(f"==============================================================\n")