
import asyncio
import random
import numpy as np
import time
import threading
import openai
import logging
import os
import json
import logging
import re
import argparse
import random
import jsonlines
from datetime import datetime

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--test_file", type=str, default='none')
parser.add_argument("--output_file", type=str, default='none')
parser.add_argument("--api_key", type=str, default='none')
parser.add_argument("--log", type=str, default='log')
args = parser.parse_args()
logging.basicConfig(filename=f'./log/{args.log}.log',    
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("openai").setLevel(logging.WARNING)



################# print log ##################################
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)
##########################################

logging.info(f"================")
logging.info(f"{args}")
logging.info(f"=============")
# print(f"{args.target}  start")

# Semaphore to limit concurrency
semaphore = asyncio.Semaphore(30)  # Adjust this number as per your desired concurrency

now_done =[0] # number of done
tokens = [0,0,0] # prompt, completion, total



async def gimme_answer(full_article):

    openai.api_key = args.api_key


    prompt=f"""[instruction] The text provided within the triple backticks (``` ''') is a Question and an Answer by an agent. contains a Question and an Answer by an agent. Your task is to evaluate whether the agent's response is factually correct or incorrect.

1) Very briefly and shortly explain whether the answer contains any factual inaccuracies.
2) Finally, classify the answer as either "True" (factually correct) or "False" (factually incorrect).

```
{full_article}
'''
"""
    result = await openai.ChatCompletion.acreate(
                    model='gpt-4o-mini-2024-07-18',
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0
                    # max_tokens = 100
                )

    answer0 = result['choices'][0]['message']['content']
    usage0 = result['usage']
    return answer0, usage0



async def test(data):
    # Simulate some asynchronous IO-bound task with asyncio.sleep
    async with semaphore:
        result = 0
        
        
        # article 번호, question 번호 받은 뒤  tmp 에 있는지 없는지 확인
        while True:
            try:
                full_article = data['question_answer'].replace('<|begin_of_text|>','').replace('<|eot_id|>','') + '...'
                answer, usage = await gimme_answer(full_article)

                now_done[0]+=1
                tokens[0] += usage['prompt_tokens']
                tokens[1] += usage['completion_tokens']
                tokens[2] += usage['total_tokens']

                data['geval_raw'] = answer
                data['question_answer'] = full_article
                q_id = data['q_id']
                logging.info(f"{q_id} Done  ({now_done[0]})")
                break

            except:
                logging.exception(f"error")
                await asyncio.sleep(10)
        return data


async def main():
    # Using asyncio.gather to get results as they come
    with jsonlines.open(f"{args.test_file}", 'r') as reader:
        datas = list(reader)#[:10]
    result = await asyncio.gather(*(test(data) for data in datas))
    
    return result

# Run the asyncio event loop
loop = asyncio.get_event_loop()
all_results_temp = loop.run_until_complete(main())

print(len(all_results_temp))
print(all_results_temp[0])

output = args.test_data.replace('d2_datasets_answered','d3_datasets_k_labeled')

with jsonlines.open(f"{output}", 'w') as writer:
    writer.write_all(all_results_temp)

logging.info(f"Total : {len(all_results_temp)}")
logging.info(f"token : prompt {tokens[0]} / compl {tokens[1]}/ tot {tokens[2]}")
logging.info(f"===========  end==============")