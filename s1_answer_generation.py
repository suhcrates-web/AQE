import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
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
parser.add_argument('--max_new_tokens', type=int, default=100)
parser.add_argument('--token', type=str, default='your_token')
parser.add_argument('--cache', type=str, default='your_cache')
parser.add_argument('--randint', type=int, default=1111)
parser.add_argument('--masterport', type=str, default='12345')
args = parser.parse_args()


args.model_title = args.model_name.split('/')[-1]


#####################
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.masterport

    # 작업 그룹 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

#################################

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
    
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


    def forword(self, **kwargs):
        return self.model(**kwargs)
    
    def generate(self, **kwargs):
        return self.model.generate(**kwargs, do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id = self.tokenizer.eos_token_id)


def demo_basic(rank, world_size, test_data): # run()
# def extract_information(test_data, filtered_indices): # run()
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


    dataset=[]
    total_steps = len(dataloader_test) 
    for step, batch_data in tqdm(enumerate(dataloader_test) , total=total_steps):
        question = batch_data['prompt']
        tokens = tokenizer(question, return_tensors='pt', truncation=True, padding=True).to('cuda')
        result = model.generate(**tokens )  
        # print(result)              
        
        for i, q_id in enumerate(batch_data['q_id']):
            dic ={}


            if type(q_id) == torch.Tensor:
                q_id = q_id.tolist()
            dic['q_id'] = q_id#.tolist()
            dic['question'] = batch_data['question'][i]
            dic['prompt'] =batch_data['prompt'][i]
            dic['question_answer'] = tokenizer.decode(result[i] ,skip_special_tokens=True)
            dic['answer'] = dic['question_answer'][len(dic['prompt']):]
            
            if 'answerType' in batch_data:
                dic['answerType'] = batch_data['answerType'][i]
            if 'category' in batch_data:
                dic['category'] = batch_data['category'][i]
            if 'complexityType' in batch_data:
                dic['complexityType'] = batch_data['complexityType'][i]

            if 'label' in batch_data:
                dic['label'] = batch_data['label'][i]

            dataset.append(dic)
    dist.barrier()
    torch.save(dataset, f"tmp/{args.randint}_{rank}")
    return dataset
        
def run_demo(demo_fn, world_size, test_data):
    mp.spawn(demo_fn,  
            args=(world_size, test_data),
            nprocs=world_size,
            join=True)

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count() 
    world_size = n_gpus
    
    test_data = load_dataset(path= os.path.dirname(args.test_data),data_files=[os.path.basename(args.test_data)])['train']

    def qa_sentence(example):
        example['prompt'] = f"[Question]:{example['question']} [Answer]:"
        return example
    test_data = test_data.map(qa_sentence)

    with mp.Manager() as manager:

        run_demo(demo_basic, world_size, test_data)

        generated_results = []
        for i in range(world_size):
            shard = torch.load(f"tmp/{args.randint}_{i}")
            generated_results.extend(shard)
            os.remove(f"tmp/{args.randint}_{i}")
        dics = {}
        
    for line in generated_results:  #중복제거
        dics[line['q_id']] = line  
    sorted_values = [value for key, value in sorted(dics.items())]  # 정렬

    print(len(generated_results))
    print(generated_results[0])

    torch.save(sorted_values, f"{args.output}")