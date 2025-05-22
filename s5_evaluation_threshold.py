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
import json


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='none')
parser.add_argument('--benchmark', type=str, default='none') 
parser.add_argument('--method', type=str, default='scao_thre') 
#scao_thre, somewords_thre
parser.add_argument('--train_data', type=str, default='') 
parser.add_argument('--dev_data', type=str, default='') 
parser.add_argument('--test_data', type=str, default='') 
parser.add_argument('--k_label', type=str, default='fok_naive') 
parser.add_argument('--output', type=str, default='') 
args = parser.parse_args()
###### 
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
    for data in dataset:
        del data['hidds_somewords']   
        del data['hidds_oneword']
    test_data = Dataset.from_list(dataset)
    test_data.set_format(type='torch')
    
    
    dataset = torch.load(args.train_data)
    for data in dataset:
        del data['hidds_somewords']
        del data['hidds_oneword']
    train_data = Dataset.from_list(dataset)
    train_data.set_format(type='torch')

    dataset = torch.load(args.dev_data)
    for data in dataset:
        del data['hidds_somewords']
        del data['hidds_oneword']
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


########
if args.method in ['conf_scao']:
    add_hidds='_oneword'
elif args.method in ['conf']:
    add_hidds='_somewords'
else:
    raise Exception("unexpected method")

num_layers = train_data[0][f'hidds{add_hidds}'].shape[0]
dim_layers = train_data[0][f'hidds{add_hidds}'].shape[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


####### Load Model #########


dics = {}
for data in ['oneword_probs', 'somewords_probs']:
    dics[data] = {}
    
    lines = list(train_data)


    max_acc_list = []
    k_list = []
    max_th_list = []
    for k in range(1,31):
        X= []
        Y= []
        for line in lines:

            x = np.mean(line[data][:k])
            y=1 if line['k_label'] else 0
            X.append(x)
            Y.append(y)
            
        if 'logits' in data:
            scale0 = (-100,100,1000)
        elif 'probs' in data:
            scale0 = (0,0.1, 3000)
        
        th_list = []
        acc_list = []
        for th in np.linspace(*scale0):
            acc = np.equal(X > th, Y).mean()
            acc_list.append(acc)
            th_list.append(th)
        max_acc = np.max(acc_list)
        max_th = th_list[np.argmax(acc_list)]
        max_th_list.append(max_th)
        max_acc_list.append(max_acc)
        k_list.append(k)
    _max_acc = np.max(max_acc_list)
    _max_th = max_th_list[np.argmax(max_acc_list)]
    _max_k = k_list[np.argmax(max_acc_list)]


    print(f"{data} {_max_acc} {_max_th} {_max_k}")

    dics[data]={'max_k': _max_k, 'max_acc':_max_acc, 'max_th':_max_th, }
    

#### test phase
dic = {}

print("Eval result")
print("  accuracy,  auroc")
for data in ['oneword_probs', 'somewords_probs']:    
        lines= list(test_data)

        X=[]
        Y=[]
        for line in lines:
            x = np.mean(line[data][:dics[data]['max_k']])
            y=1 if line['k_label'] else 0
            X.append(x)
            Y.append(y)

        acc = np.equal(X > dics[data]['max_th'], Y).mean()
        auroc = roc_auc_score(Y, X)
        print(f"{data} {acc}, {auroc}")
        dic[data] = {'acc': acc.tolist(), 'auroc': auroc}
        



print(dic)
with open(f"results/hallu/{args.output}_{args.method}_{args.probs}.json", 'w') as f:
    json.dump(dic, f, indent=4)
    