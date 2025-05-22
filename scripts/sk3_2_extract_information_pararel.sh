# original
python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/ParaRel/original/pararel_ID_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/ParaRel/original/pararel_ID_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"

python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/ParaRel/original/pararel_OOD_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/ParaRel/original/pararel_OOD_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"


python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d4_datasts_k_refined/ParaRel/original/pararel_training data_Meta-Llama-3-8B-Instruct.jsonl" \
--output="./datasets/d5_datasts_k_processed/ParaRel/original/pararel_training data_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"