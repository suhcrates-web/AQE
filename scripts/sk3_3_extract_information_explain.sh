# original

python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/Explain/original/explain_train_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/Explain/original/explain_train_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"

python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/Explain/original/explain_dev_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/Explain/original/explain_dev_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"


python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/Explain/original/explain_test_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/Explain/original/explain_test_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"


# type

python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/Explain/type/explain_train_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/Explain/type/explain_train_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"

python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/Explain/type/explain_dev_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/Explain/type/explain_dev_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"


python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/Explain/type/explain_test_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/Explain/type/explain_test_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"