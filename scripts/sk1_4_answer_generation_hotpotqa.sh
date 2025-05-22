python s1_answer_generation.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d1_datasets/HotpotQA/hotpot_10k.jsonl" \
--output="./datasets/d2_datasets_answered/HotpotQA/hotpot_10k_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"

python s1_answer_generation.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d1_datasets/HotpotQA/hotpot_test.jsonl" \
--output="./datasets/d2_datasets_answered/HotpotQA/hotpot_test_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"