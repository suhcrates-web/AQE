import argparse
import os
import torch
import jsonlines
parser = argparse.ArgumentParser()

parser.add_argument('--test_data', type=str, default='none')
# parser.add_argument('--output', type=str, default='none')

parser = argparse.ArgumentParser()
args = parser.parse_args()

dataset =[]
with jsonlines.open(args.test_data, 'r') as f:
    lines = list(f)

    for line in lines:
        if line['labe'].lower() in line['answer'].lower():
            line['k_label'] = True
        else:
            line['k_label'] = False
        dataset.append(line)

output = args.test_data.replace('d2_datasets_answered','d3_datasets_k_labeled')
print(f"save into {output}")


with jsonlines.open(output, 'w') as writer:
    writer.write_all(dataset)