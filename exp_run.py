import argparse
import yaml
import wandb
from dataset import load_dataset,TUData,OGB_Data
from train  import train_model
from utils import seed_everything, record_result
import os
import torch
from train_dist  import data_loader
from ogb.graphproppred import Evaluator

parser = argparse.ArgumentParser(description='  ')

parser.add_argument('--model', default='GMT', type=str)
parser.add_argument('--dataset', default='ogbg-molhiv', type=str,help='on which dataset')
parser.add_argument('--device', required=False, default=7, type=int, help='Device Number' )
parser.add_argument("--seed", type=int, default=1234, help="random seed (default: 1234)")
parser.add_argument("--repeat_time", type=int, default=10)
parser.add_argument("--wandb_record", action="store_true")
args = parser.parse_args()

seed_everything(args.seed)

config_path="./config/"+args.model+".yaml"
config=yaml.safe_load(open(config_path,'r'))


wandb_config = {
    "epochs": int(config[args.dataset]["epochs"]),
    "hidden_size":int(config[args.dataset]["hidden_size"]),
    "learning_rate":float(config[args.dataset]["learning_rate"]),
    "dropout":float(config[args.dataset]["dropout"]),
    "batch_size":int(config[args.dataset]["batch_size"]),
    "num_layers":int(config[args.dataset]["num_layers"]),
    "weight_decay":float(config[args.dataset]["weight_decay"]),
}

for key in config[args.dataset].keys():
    wandb_config[key] = config[args.dataset][key]


#only save data split in training
save_data=True

if __name__ == '__main__':
    test_acc_list = []
    for item in range(args.repeat_time):
        project_name=args.model+'_'+args.dataset+"_model_repeat"

        dataset=load_dataset(args.dataset,args.model,shuffle=True,)
        dataloaders = data_loader(args.dataset, dataset, wandb_config['batch_size'])

        if save_data:
            folder="./model&dataset/"+args.dataset+"/"+args.model+'/'+'without_loss'
            if not os.path.exists(folder):
                os.makedirs(folder)
            torch.save(dataset, os.path.join(folder, f'splited_{item}.pt'))
            print("Shuffled Data saved to ",folder)

        if args.wandb_record:
            run = wandb.init(project=project_name,name="run"+str(item),config=wandb_config)

        test_acc_record=train_model(args.model,
                        dataset,
                        dataloaders=dataloaders,
                        config=wandb_config,
                        device=args.device,
                        wandb_record=args.wandb_record,
                        save_model=True,
                        patience=30,#wandb.config.patience,
                        min_delta=0.005,#wandb.config.patience,
                        seed=args.seed,
                        fromstore=False,
                        evaluator=Evaluator(args.dataset) if args.dataset in OGB_Data else None,
                        task_type=dataset.task_type if args.dataset in OGB_Data else None,
                        repeat_time=item, 
                        )     
        test_acc_list.append(test_acc_record)
        
        
        if args.wandb_record:
            wandb.finish()
          
    test_acc_list = torch.tensor(test_acc_list)
    res = f'{test_acc_list.mean()*100:.2f}±{test_acc_list.std()*100:.2f}'
    print(f'Model: {args.model}, Dataset: {args.dataset}, Performance: {res}') 
    
    record_result(args.model, args.dataset, f'{res}', root='./results.csv')

