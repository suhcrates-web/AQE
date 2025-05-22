python s1_answer_generation.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d1_benchmarks/ParaRel_OOD/pararel_ID.jsonl" \
--output="./datasets/d2_datasets_answered/ParaRel_OOD/pararel_ID_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"

python s1_answer_generation.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d1_benchmarks/ParaRel_OOD/pararel_OOD.jsonl" \
--output="./datasets/d2_datasets_answered/ParaRel_OOD/pararel_OOD_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"


python s1_answer_generation.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d1_benchmarks/ParaRel_OOD/pararel_training_data.jsonl" \
--output="./datasets/d2_datasets_answered/ParaRel_OOD/pararel_training_data_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"