from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import argparse
import json
from transformers import BertConfig, BertForMaskedLM, BertTokenizer, BertForSequenceClassification, AutoTokenizer , BertModel
import numpy as np
from utils.tools import inject_param


parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, default='none') # train_split, split, asone
parser.add_argument('--train_data', type=str, default='') 
parser.add_argument('--dev_data', type=str, default='') 
parser.add_argument('--test_data', type=str, default='')  
parser.add_argument('--output', type=str, default='')
args = parser.parse_args()
###### 


##### Load dataset ######

if args.benchmark == 'asone':
    dataset = load_dataset('json',data_files=args.train_data , split='train')
    train_valid_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_valid = train_valid_test['train'].train_test_split(test_size=0.25, seed=42)

    train_data = train_valid['train']
    dev_data = train_valid['test']  # This is the validation set
    test_data = train_valid_test['test']  

    train_data.set_format(type='torch')
    dev_data.set_format(type='torch')
    test_data.set_format(type='torch')
    
elif args.benchmark == 'split':
    train_data = load_dataset('json',data_files=args.train_data , split='train')
    train_data.set_format(type='torch')

    dev_data = load_dataset('json',data_files=args.dev_data , split='train')
    dev_data.set_format(type='torch')
    
    test_data = load_dataset('json',data_files=args.test_data , split='train')
    test_data.set_format(type='torch')

elif args.benchmark == 'train_split':

    dataset = load_dataset('json',data_files=args.train_data , split='train')
    train_valid = dataset.train_test_split(test_size=0.2, seed=42)

    train_data = train_valid['train']
    dev_data = train_valid['test']  # This is the validation set


    test_data = load_dataset('json',data_files=args.test_data , split='train')
    test_data.set_format(type='torch')

    train_data.set_format(type='torch')
    dev_data.set_format(type='torch')
    test_data.set_format(type='torch')

if 'qa_seq' in train_data.column_names:
    train_data = train_data.remove_columns(["qa_seq"])
    dev_data = dev_data.remove_columns(["qa_seq"])
    test_data = test_data.remove_columns(["qa_seq"])

print(f"Data loaded : {args.benchmark}")
train_loader = DataLoader(train_data,shuffle=True ,batch_size=16)
test_loader = DataLoader(test_data,shuffle=False ,batch_size=16)
dev_loader = DataLoader(dev_data,shuffle=False ,batch_size=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


####### Load Model #########

class Model(nn.Module):
    def __init__(self,  hidden_size, compress_size):
        super(Model, self).__init__()  # 이거 없음 안됨
        self.sbert = BertModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2").requires_grad_(False)
        self.compression = nn.Linear(self.sbert.config.hidden_size, compress_size)   

        self.fc1 = nn.Linear(compress_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.final = nn.Linear(hidden_size, 1)


    def forward(self, query):
        ##
        with torch.no_grad():
            pooled = self.sbert(**query).pooler_output
        compressed = self.compression(pooled)

        x= compressed

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y_predicted = torch.sigmoid(self.final(x))
        return y_predicted


compress_size = 10
hidden_size = 40
model = Model(hidden_size, compress_size).to(torch.float32).to(0)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()


#### eval function

def eval(model, dataloader0):
    with torch.no_grad():
        correct_sum =0
        preds = []
        labels = []
        for batch in dataloader0:
            # confidence = torch.stack(batch[data]).T.to(torch.float32).to(7)
            # confidence = confidence[:,:k] 

            tokens = tokenizer(batch['question'], return_tensors='pt', padding=True, truncation=True).to(0)

            pred = model(query=tokens)

            preds.extend(pred.cpu())
            labels.extend(batch[args.fok_label].float().cpu())

            pred_binary = (pred >= 0.5).float()
            label = batch[args.fok_label].float().to(0).unsqueeze(-1)
            correct_sum += (pred_binary == label).sum()
            # break

        auroc = roc_auc_score(labels, preds)

        acc = correct_sum / dataloader0.dataset.num_rows
    return acc.cpu(), auroc
    

#### training phase
max_epochs=10
loss_list = []
dev_acc_list = []
test_acc_list = []
param_list = []
name_list = []
dev_acc, auroc= eval(model, dev_loader)
dev_acc_list.append(dev_acc)
print(f"{dev_acc}")
for epoch in range(max_epochs):
    for batch in train_loader:
        tokens = tokenizer(batch['question'], return_tensors='pt', padding=True, truncation=True).to(0)

        pred = model(query=tokens)
        loss = criterion(pred,  batch[args.fok_label].float().to(0).unsqueeze(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        loss_list.append(loss.detach().to('cpu'))

    dev_acc, auroc= eval(model, dev_loader)
    test_acc, auroc= eval(model, test_loader)
    dev_acc_list.append(dev_acc)
    print(f"dev_acc: {dev_acc}")
    params= []
    name_list = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(param)
            name_list.append(name)
    param_list.append(params)


#### choose best parameter
best_param=param_list[np.argmax(dev_acc_list[1:])]
inject_param(model, best_param, name_list, 'detach_reqgrad')


##### AME score
acc_list = []
auroc_list = []
for _ in range(1):
    acc, auroc = eval(model, test_loader)
    print(f"acc: {acc} , auroc : {auroc}")
    acc_list.append(acc.tolist())
    auroc_list.append(auroc)

dic = {'acc':acc_list , 'auroc_list':auroc_list}
with open(args.output, 'w') as f:
    json.dump(dic, f, indent=4)