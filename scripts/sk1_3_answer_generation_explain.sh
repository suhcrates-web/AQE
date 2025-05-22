python s1_answer_generation.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d1_benchmarks/Explain/explain_train.jsonl" \
--output="./datasets/d2_datasets_answered/Explain/explain_train_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"

python s1_answer_generation.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d1_benchmarks/Explain/explain_dev.jsonl" \
--output="./datasets/d2_datasets_answered/Explain/explain_dev_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"


python s1_answer_generation.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d1_benchmarks/Explain/explain_test.jsonl" \
--output="./datasets/d2_datasets_answered/Explain/explain_test_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"