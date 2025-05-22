import argparse
import re
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument('--test_data', type=str, default='none')
args = parser.parse_args()


file_name = args.test_data.split('/')[-1]
benchmark = args.test_data.split('/')[-2]

with jsonlines.open(args.test_data, 'r') as f:
    lines = list(f)

with jsonlines.open(f"/rtuning/datasets/{benchmark}/{file_name}", 'w') as writer:


    for line in lines:
        if len(line['label'].split(','))>1:
            continue
        label = line['label']
        a_text = line['a_text']
        a_text = re.sub(r'\nAnswer:','. ',a_text)

        sure = "sure" if line['k_label'] else "unsure"
        line['qa_verify_max50']= f"{line['q_text']} {line['a_text'].strip()}... [Verify] Are you sure you accurately answered the question based on your internal knowledge? I am "#{sure}"

        yesno = "yes" if line['k_label'] else "no"

        line['q_only_verify'] = f"Question: {line['question']} [Verify] Do you have enough internal knowledge to answer this question? "#{yesno}"
        writer.write(line)

