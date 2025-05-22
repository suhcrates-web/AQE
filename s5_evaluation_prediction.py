from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import argparse
import numpy as np
import random
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='none')
parser.add_argument('--benchmark', type=str, default='none') 
parser.add_argument('--method', type=str, default='scao_probe') # scao_probe, probe_linear, probe_dnn, scao_probe_normalize
parser.add_argument('--train_data', type=str, default='') 
parser.add_argument('--dev_data', type=str, default='') 
parser.add_argument('--test_data', type=str, default='') 
parser.add_argument('--output', type=str, default='') 
parser.add_argument('--seed', type=int, default=41) 
args = parser.parse_args()
###### 

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
model_title = args.model_name.split('/')[-1]


##### Load dataset ######

if args.benchmark == 'asone':
    dataset = torch.load(args.train_data)
    datset = Dataset.from_list(dataset)
    train_valid_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_valid = train_valid_test['train'].train_test_split(test_size=0.25, seed=42)

    train_data = train_valid['train']
    dev_data = train_valid['test']  # This is the validation set
    test_data = train_valid_test['test']  

    train_data.set_format(type='torch')
    dev_data.set_format(type='torch')
    test_data.set_format(type='torch')
    
elif args.benchmark == 'split':
    dataset = torch.load(args.test_data)
    test_data = Dataset.from_list(dataset)
    test_data.set_format(type='torch')
    
    
    dataset = torch.load(args.train_data)
    train_data = Dataset.from_list(dataset)
    train_data.set_format(type='torch')

    dataset = torch.load(args.dev_data)
    dev_data = Dataset.from_list(dataset)
    dev_data.set_format(type='torch')



elif args.benchmark == 'train_split':

    dataset = torch.load(args.train_data)
    dataset = Dataset.from_list(dataset)
    train_valid = dataset.train_test_split(test_size=0.2, seed=42)

    train_data = train_valid['train']
    dev_data = train_valid['test']  # This is the validation set


    dataset = torch.load(args.test_data)
    test_data = Dataset.from_list(dataset)

    train_data.set_format(type='torch')
    dev_data.set_format(type='torch')
    test_data.set_format(type='torch')

print(f"Data loaded : {args.benchmark}")


train_loader = DataLoader(train_data,shuffle=True ,batch_size=16)
test_loader = DataLoader(test_data,shuffle=False ,batch_size=16)
dev_loader = DataLoader(dev_data,shuffle=False ,batch_size=16)

########
if args.method in ['probe_conf_scao']:
    add_hidds='_oneword'
elif args.method in ['probe_dnn', 'probe_conf']:
    add_hidds='_somewords'
else:
    raise Exception("unexpecte method")

num_layers = train_data[0][f'hidds{add_hidds}'].shape[0]
dim_layers = train_data[0][f'hidds{add_hidds}'].shape[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


####### Load Model #########


# Model
if 'probe_conf' in args.method :
    class Model(nn.Module):
        def __init__(self, n_input_features):
            super(Model, self).__init__()  
            self.fc1 = nn.Linear(n_input_features, 30)
            self.fc2 = nn.Linear(60, 60)
            self.final = nn.Linear(60, 1, bias=True)

        def forward(self, x, conf):  # conf =batch['oneword_probs']
            x = F.relu(self.fc1(x))
            x = torch.concat((x,conf),dim=-1)
            x = F.relu(self.fc2(x))
            y_predicted = torch.sigmoid(self.final(x))
            return y_predicted



elif args.method == 'probe_dnn':
    class Model(nn.Module):
        def __init__(self, n_input_features):
            super(Model, self).__init__() 
            self.fc1 = nn.Linear(n_input_features, 60)
            self.fc2 = nn.Linear(60, 60)
            self.final = nn.Linear(60, 1, bias=True)

        def forward(self, x, _):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            y_predicted = torch.sigmoid(self.final(x))
            return y_predicted

else:
    raise Exception("unexpected method")


models = []
optimizers = []
for i in range(num_layers): ## 33
    models.append(Model(dim_layers).to(0)) 
    optimizers.append(torch.optim.AdamW(models[i].parameters()))

criterion = nn.BCELoss()


#### eval function
def eval_dev_test(models, dev_loader, test_loader):
    with torch.no_grad():
        accs = torch.zeros(num_layers).to(0)
        losses = torch.zeros(num_layers).to(0)
        for batch in dev_loader:
            for i in range(num_layers):
                pred = models[i](batch[f'hidds{add_hidds}'][:,i,:].to(0), batch['oneword_probs'].to(0)).squeeze()

                accs[i] += torch.eq(pred >0.5 , batch['k_label'].to(0)).sum()

                loss = criterion(pred, batch['k_label'].float().to(0))
                losses[i] += loss

        dev_accs = accs / dev_data.num_rows
        dev_losses = losses / dev_data.num_rows

        max_layer = torch.argmax(dev_accs)
        acc= 0
        loss= 0
        preds = []
        labels = []
        for batch in test_loader:
            pred = models[max_layer](batch[f'hidds{add_hidds}'][:,max_layer,:].to(0), batch['oneword_probs'].to(0)).squeeze()

            preds.extend(pred.cpu())
            labels.extend(batch['k_label'].float().cpu())
            acc += torch.eq(pred >0.5 , batch['k_label'].to(0)).sum()
            loss += criterion(pred, batch['k_label'].float().to(0))

        acc = acc / test_data.num_rows
        loss = loss / test_data.num_rows

        auroc = roc_auc_score(labels, preds)

        return dev_accs, dev_losses, acc, loss, auroc
    





accs, losses, test_acc, test_loss, auroc = eval_dev_test(models, dev_loader, test_loader)
print(f"before start:  max accs: {test_acc}, auroc:{auroc} on layer {torch.argmax(accs)}, dev acc {torch.max(accs)}")

for e, epoch in enumerate(range(20)):
    for step, batch in tqdm(enumerate(train_loader)):
        for i in range(num_layers):
            pred = models[i](batch[f'hidds{add_hidds}'][:,i,:].to(0), batch['oneword_probs'].to(0)).squeeze()
            loss = criterion(pred, batch['k_label'].float().to(0))
            loss.backward()
            optimizers[i].step()
            optimizers[i].zero_grad()
                
    accs, losses, test_acc, test_loss, auroc = eval_dev_test(models, dev_loader, test_loader)
    print(f"{e}  acc: {test_acc :.6f} , auroc: {auroc :.6f} , dev acc {torch.max(accs)}")
    

print(f"acc: {test_acc :.6f} , auroc: {auroc :.6f} , dev acc {torch.max(accs)}")


dic = {'test_acc':test_acc , 'auroc':auroc, 'loss':test_loss}
print(dic)
with open(f"results/hallu/{args.output}_{args.method}_{args.probs}.json", 'w') as f:
    json.dump(dic, f, indent=4)
    