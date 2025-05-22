# original
python s4_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/HotpotQA/original/hotpot_test_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/HotpotQA/original/hotpot_test_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"


python s4_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/HotpotQA/original/hotpot_10k_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/HotpotQA/original/hotpot_10k_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"


# type
python s4_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/HotpotQA/type/hotpot_test_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/HotpotQA/type/hotpot_test_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"


python s4_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/HotpotQA/type/hotpot_10k_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/HotpotQA/type/hotpot_10k_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"
