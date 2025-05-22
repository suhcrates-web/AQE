## mintaka
# original

python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/Mintaka/original/mintaka_train_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/Mintaka/original/mintaka_train_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"

python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/Mintaka/original/mintaka_dev_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/Mintaka/original/mintaka_dev_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"


python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/Mintaka/original/mintaka_test_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/Mintaka/original/mintaka_test_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"


# type
python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/Mintaka/type/mintaka_train_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/Mintaka/type/mintaka_train_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"

python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/Mintaka/type/mintaka_dev_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/Mintaka/type/mintaka_dev_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"


python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/Mintaka/type/mintaka_test_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/Mintaka/type/mintaka_test_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"



# type + domain
python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/Mintaka/type_domain/mintaka_train_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/Mintaka/type_domain/mintaka_train_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"

python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/Mintaka/type_domain/mintaka_dev_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/Mintaka/type_domain/mintaka_dev_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"


python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/Mintaka/type_domain/mintaka_test_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/Mintaka/type_domain/mintaka_test_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"
