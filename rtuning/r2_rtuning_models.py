
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoTokenizer,AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model 
from datasets import load_dataset
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
import pickle 
from torch.utils.data import DataLoader, Sampler
from collections import OrderedDict
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='none')
parser.add_argument('--benchmark', type=str, default='none') 
parser.add_argument('--gpu_train_batch_size', type=int, default=8) # 보통 4로 함
parser.add_argument('--train_grad_accum', type=bool, default=False)
parser.add_argument('--train_grad_accum_step', type=int, default=1)
parser.add_argument('--text_key', type=str, default="none")#qa_verify_max50, q_only_verify

parser.add_argument('--max_epochs', type=int, default=1)
parser.add_argument('--output_dir', type=str, default='none') 
parser.add_argument('--train_data', type=str, default='none') 
parser.add_argument('--eval_on', type=str, default='epoch') 
parser.add_argument('--eval_step', type=int, default=100) 
parser.add_argument('--eval_with_save', type=bool, default=False) 

parser.add_argument('--token', type=str, default="none") 
parser.add_argument('--cache', type=str, default="none") 

args = parser.parse_args()




class Model_qlora(nn.Module):
    def __init__(self, args):
        super(Model_qlora, self).__init__()
        self.args = args

        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(args.model_name,
        token=args.token, cache_dir=args.cache,
        quantization_config=bnb_config
        )#.half()
        self.model.config.use_cache = False
        lora_alpha = 16
        lora_dropout = 0.1
        lora_r = 64   #

        peft_config = LoraConfig(
        lora_alpha = lora_alpha,
        lora_dropout= lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",)
        self.model = get_peft_model(self.model, peft_config)

        for name, param in self.model.named_parameters():
            if param.dtype == torch.bfloat16:
                param.to(torch.float32)
                param.data = param.data.to(torch.float32)

    def forward(self, **kwargs):
  
            return self.model(**kwargs).loss





def eval(dataloader, model, tokenizer, sure,unsure):

    with torch.no_grad():
        loss_list = []
        loss_count = torch.tensor(0, device='cuda')
        for batch_data in tqdm(dataloader ):
            # print(batch_data)
            temp= batch_data[args.text_key]
            texts = []
            for i, text in enumerate(temp):
        
                yesno = sure if batch_data['k_label'][i] else unsure
                texts.append(text + yesno)  
                    
            tokens = tokenizer(texts, return_tensors='pt', truncation=True, padding=True).to('cuda')
            tokens['labels'] = tokens.input_ids

            loss = model(**tokens)
            loss_list.append(loss)
            loss_count += tokens.input_ids.shape[0]
        loss_mean = torch.stack(loss_list).detach().sum()
    return loss_mean/loss_count



def save_checkpoint(model, memo):
    with torch.no_grad():
        dic=OrderedDict()

        for name, param in model.named_parameters():
            if param.requires_grad:
                dic[name] = param.detach().to('cpu')
        torch.save(dic, f"./model_output/{args.output_dir}/checkpoint_{memo}" )

history = {'loss':[], 'loss_mean':[], 'eval_loss':[], 'eval_step':[], 'config':{**args.__dict__}}


def main(train_data, test_data):
    train_sampler = Sampler(train_data, shuffle=True, seed=42)
    train_dataloader = DataLoader(train_data, batch_size = args.gpu_train_batch_size,  sampler=train_sampler)


    test_dataloader = DataLoader(test_data,shuffle=False ,batch_size=16)


    model = Model_qlora(args)

    ############

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.token, cache_dir=args.cache)

    tokenizer.pad_token = tokenizer.eos_token


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.zero_grad()


    if args.text_key == 'qa_verify_max50':
        sure="sure"
        unsure = "unsure"
    elif args.text_key == 'q_only_verify':
        sure="yes"
        unsure = "no"
    else:
        raise Exception("unexpected   text_key")
    

    eval_loss = eval(test_dataloader, model, tokenizer, sure,unsure)
    dist.barrier()

    print(f"before start / eval Loss : {eval_loss}")
    history['eval_loss'].append(eval_loss.to('cpu'))
    history['eval_step'].append(0)



    for epoch in range(1, args.max_epochs+1):  
        # model.train()

        loss_list=[]
        for step, (batch_data) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # qa_verify_max50, q_only_verify
            temp= batch_data[args.text_key]
            texts = []
            for i, text in enumerate(temp):
        
                yesno = sure if batch_data['k_label'][i] else unsure
                texts.append(text + yesno)  
                    
            tokens = tokenizer(texts, return_tensors='pt', truncation=True, padding=True).to('cuda')
            tokens['labels'] = tokens.input_ids
            loss = model(**tokens)
            loss = loss / args.train_grad_accum_step # default 1

            loss.backward()
            loss_list.append(loss.clone().detach())

            # gradiant accumulation # no accum   when  accum_step=1
            if ((step + 1) % args.train_grad_accum_step ==0) or (step + 1) == len(train_dataloader) :
                optimizer.step()
                optimizer.zero_grad()
            
            dist.barrier()

            if args.eval_on=='step' and  ((step + 1) % args.eval_step ==0) :

                if args.eval_with_save:
                    save_checkpoint(model, f"{epoch}_{step + 1}")
                

                with torch.no_grad():
                    loss_count = torch.tensor(len(loss_list))
                    loss_mean = torch.stack(loss_list).detach().sum()
                    loss_list = []

                    loss_mean = loss_mean/loss_count

                    eval_loss = eval(test_dataloader, model, tokenizer, sure,unsure)

                    history['loss_mean'].append(loss_mean.to('cpu'))

                    history['eval_loss'].append(eval_loss.to('cpu'))
                    history['eval_step'].append(f"{epoch}_{step + 1}")
                    with open(f"./model_output_yb/{args.output_dir}/history.pkl", 'wb') as f:
                        pickle.dump(history, f)

                    print(f"{epoch}_{step +1} / Loss : {loss_mean} / eval Loss : {eval_loss}") ## loss.item() : loss 
                    

        
        with torch.no_grad():
            loss_count = torch.tensor(len(loss_list))
            loss_mean = torch.stack(loss_list).detach().sum()


            loss_mean = loss_mean/loss_count

            eval_loss = eval(test_dataloader, model, tokenizer,  sure, unsure)

            history['loss_mean'].append(loss_mean.to('cpu'))

            history['eval_loss'].append(eval_loss.to('cpu'))
            history['eval_step'].append(f"{epoch}_e")
            print(f"{epoch} epoch / Loss : {loss_mean} / eval Loss : {eval_loss}") ## loss.item() : loss 
            


        save_checkpoint(model, f"{epoch}_e")





 

if __name__ == "__main__":
   

    model_title = args.model_name.split('/')[-1]
    if args.benchmark=='pararel':
        train_file = f"/rtuning/datasets/ParaRelOOD/training_data_{model_title}_somewords_token50.jsonl"
        dev_file = f"/rtuning/datasets/ParaRelOOD/ID_test_{model_title}_somewords_token50.jsonl"
        test_file = f"/rtuning/datasets/ParaRelOOD/OOD_test_{model_title}_somewords_token50.jsonl"

        train_data = load_dataset(path= os.path.dirname(train_file),data_files=[os.path.basename(train_file)])['train']
        dev_data = load_dataset(path= os.path.dirname(dev_file),data_files=[os.path.basename(dev_file)])['train']
        test_data = load_dataset(path= os.path.dirname(test_file),data_files=[os.path.basename(test_file)])['train']    
    
    elif args.benchmark=='mintaka':

        train_file = f"/rtuning/datasets/Mintaka/mintaka_train_{model_title}.jsonl"
        dev_file = f"/rtuning/datasets/Mintaka/mintaka_dev_{model_title}.jsonl"
        test_file = f"/rtuning/datasets/Mintaka/mintaka_test_{model_title}.jsonl"

        train_data = load_dataset(path= os.path.dirname(train_file),data_files=[os.path.basename(train_file)])['train']
        dev_data = load_dataset(path= os.path.dirname(dev_file),data_files=[os.path.basename(dev_file)])['train']
        test_data = load_dataset(path= os.path.dirname(test_file),data_files=[os.path.basename(test_file)])['train']
   

    elif args.benchmark == 'haluevalQA':
        train_file = f"/rtuning/datasets/HaluEval/HaluEvalQA_{model_title}.jsonl"
        dataset = load_dataset(path= os.path.dirname(train_file),data_files=[os.path.basename(train_file)])['train']
        train_valid_test = dataset.train_test_split(test_size=0.2, seed=1)
        train_valid = train_valid_test['train'].train_test_split(test_size=0.25, seed=1)
        train_data = train_valid['train']
        dev_data = train_valid['test']  # This is the validation set
        test_data = train_valid_test['test']  
    
    elif args.benchmark == 'hotpotQA':
        train_file =f'/rtuning/datasets/HotpotQA/hotpot_10k_{model_title}.jsonl'
        test_file =f'/rtuning/datasets/HotpotQA/hotpot_test_{model_title}.jsonl'

        train_data =  load_dataset(path= os.path.dirname(train_file),data_files=[os.path.basename(train_file)])['train']
        test_data =  load_dataset(path= os.path.dirname(test_file),data_files=[os.path.basename(test_file)])['train']


    print(f"train: {train_data.num_rows} / valid : {test_data.num_rows}")

    if not os.path.exists(f'/rtuning/model_output/{args.output_dir}'):
        os.makedirs(f'/rtuning/model_output/{args.output_dir}', exist_ok=True)


    main(train_data, test_data)


        