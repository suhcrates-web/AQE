import os
import sys
import tempfile
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader, Sampler, DistributedSampler
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='none')
parser.add_argument('--test_data', type=str, default='none')
parser.add_argument('--output', type=str, default='none')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--k', type=int, default=30)
parser.add_argument('--token', type=str, default='your_token')
parser.add_argument('--cache', type=str, default='your_cache')
parser.add_argument('--max_new_tokens', type=int, default=1)
parser.add_argument('--test_size', type=int, default=-1)
parser.add_argument('--k_label', type=str, default='naive_correct')
args = parser.parse_args()



args.model_title = args.model_name.split('/')[-1]


#####################
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

#################################

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.token, cache_dir=args.cache, padding=True, truncation=True, return_tensors="pt")

        bnb_config = BitsAndBytesConfig(
            # load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name,
        token=args.token, cache_dir=args.cache, output_hidden_states=True,
        quantization_config=bnb_config
        )

    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def generate(self, **kwargs):
        return self.model.generate(**kwargs, do_sample=False, max_new_tokens=self.args.max_new_tokens, pad_token_id = self.tokenizer.eos_token_id)


def demo_basic(rank, world_size, test_data): # run()
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    if rank==0:
        print(args)
           
    torch.cuda.set_device(rank) ##
    model = Model(args).to(rank)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.token, cache_dir=args.cache, padding=True, truncation=True, return_tensors="pt", padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id


    #####################


    data_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader_test = DataLoader(test_data, batch_size = args.batch_size,  sampler=data_sampler, drop_last=True)

    ##################



    with torch.no_grad():
        dataset=[]
        total_steps = len(dataloader_test) 
        for step, line in tqdm(enumerate(dataloader_test),total=total_steps):

            # q_text = f'[Question]:{line["question"][0]} You must answer in only one word. [Answer]:'
            q_text = f'You must answer in only one word. [Question]: What man was a famous American author and also a steamboat pilot on the Mississippi River? [Answer]: Mark Twain. [Question]:{line["question"][0]} [Answer]:'

            tokens = tokenizer(q_text, return_tensors='pt').to(rank)
            output = model.forward(**tokens)
            
            hidds = []
            for i, hidd in enumerate(output.hidden_states):
                # if i%5 ==0:
                    hidds.append(hidd.squeeze()[-1].squeeze())
            hidds_oneword = torch.stack(hidds).to('cpu')


            line['hidds_oneword'] = hidds_oneword

            ########

            last_token= output.hidden_states[-1][:,-1]
            logits = model.model.lm_head(last_token).squeeze()
            probs = logits.softmax(dim=-1)

            topk_logits = torch.topk(logits, k=args.k)
            topk_probs = torch.topk(probs, k=args.k)
            candidates = tokenizer.convert_ids_to_tokens(topk_logits.indices)
            line['oneword_logits'] = topk_logits.values.tolist()
            line['oneword_probs'] = topk_probs.values.tolist()
            line['oneword_candidates'] = candidates
            


            #########
            q_text = f'[Question]:{line["question"][0]} [Answer]:'

            tokens = tokenizer(q_text, return_tensors='pt').to(rank)
            output =model(**tokens)
            
            hidds = []
            for i, hidd in enumerate(output.hidden_states):
                # if i%5 ==0:
                    hidds.append(hidd.squeeze()[-1].squeeze())
            hidds_somewords = torch.stack(hidds).to('cpu')

            #######

            last_token= output.hidden_states[-1][:,-1]
            logits = model.model.lm_head(last_token).squeeze()
            probs = logits.softmax(dim=-1)

            topk_logits = torch.topk(logits, k=args.k)
            topk_probs = torch.topk(probs, k=args.k)
            candidates = tokenizer.convert_ids_to_tokens(topk_logits.indices)
            line['somewords_logits'] = topk_logits.values.tolist()
            line['somewords_probs'] = topk_probs.values.tolist()
            line['somewords_candidates'] = candidates


            line['hidds_somewords'] = hidds_somewords


            #### wrap up
            if 'id' in line:
                id = line['id']
                if type(id) == torch.Tensor:
                    id = id.tolist()
                line['id'] = id
            
            q_id = line['q_id'][0]
            if type(q_id) == torch.Tensor:
                q_id = q_id.tolist()
            line['q_id'] = q_id
            line['question'] =line['question'][0]
            if 'question_answer' in line:
                line['question_answer'] =line['question_answer'][0]
            if 'geval_raw' in line:
                line['geval_raw'] =line['geval_raw'][0]

            if 'q_text' in line:
                line['q_text'] =line['q_text'][0]
            if 'a_text' in line:
                line['a_text'] =line['a_text'][0]
            line['k_label'] =line['k_label'][0].tolist()
            

            dataset.append(line)
    dist.barrier()
    torch.save(dataset, f"tmp/{rank}")
    return dataset

def run_demo(demo_fn, world_size, test_data):
    mp.spawn(demo_fn,  # demo_fn  이  5번파일 run() 과 같음
            args=(world_size, test_data),
            nprocs=world_size,
            join=True)

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count() # 타이탄 서버는 3개
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus


    test_data = load_dataset(path= os.path.dirname(args.test_data),data_files=[os.path.basename(args.test_data)])['train']
    
    if args.test_size >=0 :
        test_data = test_data.select(range(args.test_size))
    
    def qa_sentence(example):
        if 'id' in example:
            example['q_id'] = example['id']
        return example
    test_data = test_data.map(qa_sentence)   
    
    with mp.Manager() as manager:

        run_demo(demo_basic, world_size, test_data)

        generated_results = []
        for i in range(world_size):
            shard = torch.load(f"tmp/{i}")
            generated_results.extend(shard)
            os.remove(f"tmp/{i}")
        dics = {}
        
    for line in generated_results:  
        dics[line['q_id']] = line  
    sorted_values = [value for key, value in sorted(dics.items())]  #

    print(len(generated_results))
    print(generated_results[0])

    torch.save(sorted_values, f"{args.output}")
