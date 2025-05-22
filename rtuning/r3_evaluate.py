from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig, get_peft_model 
import torch
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='none') # 
parser.add_argument('--test_data', type=str, default='none')
parser.add_argument('--adapter_file', type=str, default="")
parser.add_argument('--text_key', type=str, default="none")#qa_verify_max50, q_only_verify

parser.add_argument('--benchmark', type=str, default='none') 
parser.add_argument('--token', type=str, default='none') 
parser.add_argument('--cache', type=str, default='none') 

args = parser.parse_args()

###### function
def inject_param(model, param_list, name_list, type0):
    #type 'clone' 'clone_detach' 'detach_reqgrad' 'assign'
    for name, param in zip(name_list, param_list): 
        names = name.split('.')
        module_name = '.'.join(names[:-1])  # Get module path except for the last element
        param_name = names[-1] 
        module = model
        
        for sub_name in module_name.split('.'):
            if sub_name:
                module = getattr(module, sub_name)
        
        if type0=='clone':
            setattr(module, param_name, param.clone())
        elif type0 == 'detach_reqgrad':
            setattr(module, param_name, param.detach().requires_grad_())
        elif type0 == 'detach':
            setattr(module, param_name, param.detach())
        elif type0 == 'assign':
            setattr(module, param_name, param)


#### init #######
bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_quant_type="nf4",   # data type
bnb_4bit_compute_dtype=torch.float32,  # compute data type

)
model = AutoModelForCausalLM.from_pretrained(args.model_name, token=args.token, cache_dir=args.cache, quantization_config=bnb_config)
model.config.use_cache = False
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64   #

peft_config = LoraConfig(
lora_alpha = lora_alpha,
lora_dropout= lora_dropout,
r=lora_r,
bias="none",
task_type="CAUSAL_LM",)
model = get_peft_model(model, peft_config)

state_dict = torch.load(args.adapter_file)
name_list = [x[6:] for x in list(state_dict.keys())]   
inject_param(model, list(state_dict.values()), name_list, 'detach')
model.to(0)

for name, param in model.named_parameters():
    if param.dtype == torch.float16:
        param.to(torch.float32)
        param.data = param.data.to(torch.float32)

######

tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.token, cache_dir=args.cache)

tokenizer.pad_token = tokenizer.eos_token
#####


with jsonlines.open(args.test_data, 'r') as f:
    test_data = list(f)


if 'Meta-Llama-3-8B-Instruct' in args.model_name:
    if args.text_key == 'qa_verify_max50':
        SURE= [2771] #" sure"
        UNSURE = [44003] #" unsure"
    elif args.text_key == 'q_only_verify':
        SURE= [10035] #" yes"
        UNSURE= [912] #" no
    else:
        raise Exception("unexpected text key")

elif 'Llama-2-13b-chat-hf' in args.model_name:

    if args.text_key == 'qa_verify_max50':
        SURE= [1854] #" sure"
        UNSURE = [9644] #" unsure"
    elif args.text_key == 'q_only_verify':
        SURE= [4874] #" yes"
        UNSURE= [694] #" no
    else:
        raise Exception("unexpected text key")


with jsonlines.open(f"results/{args.benchmark}_{args.text_key}_{args.model_name.split('/')[-1]}.jsonl", 'w') as writer:
    fok_correct = 0
    for line in tqdm(test_data):
        tokens = tokenizer(line['q_only_verify'], return_tensors='pt').to(0)
        with torch.no_grad():
            output = model(**tokens)
            pt= torch.softmax(output.logits[0][-1], dim=0)

            sure_prob = pt[SURE[0]]
            unsure_prob = pt[UNSURE[0]]
            sure_prob = sure_prob/(sure_prob+unsure_prob) 
            sure_prob = sure_prob.cpu().tolist()
            line['sure_prob'] = sure_prob
            line['fok_correct'] = (sure_prob >0.5) == line['naive_correct']
            fok_correct += int(line['fok_correct'])

        
        writer.write(line)

print(f"total: {len(test_data)} / acc num : {fok_correct} / acc: {fok_correct / len(test_data)}")