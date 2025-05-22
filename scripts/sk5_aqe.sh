# mintaka original
python s6_aqe.py \
--benchmark="split" \
--train_data="datasets/d4_datasets_refined/Mintaka/original/mintaka_train_Meta-Llama-3-8B-Instruct.jsonl" \
--dev_data="datasets/d4_datasets_refined/Mintaka/original/mintaka_dev_Meta-Llama-3-8B-Instruct.jsonl" \
--test_data="datasets/d4_datasets_refined/Mintaka/original/mintaka_test_Meta-Llama-3-8B-Instruct.jsonl" \
--output="results/aqe/mintaka_test_Meta-Llama-3-8B-Instruct_original.json" 


# mintaka type
python s6_aqe.py \
--benchmark="split" \
--train_data="datasets/d4_datasets_refined/Mintaka/type/mintaka_train_Meta-Llama-3-8B-Instruct.jsonl" \
--dev_data="datasets/d4_datasets_refined/Mintaka/type/mintaka_dev_Meta-Llama-3-8B-Instruct.jsonl" \
--test_data="datasets/d4_datasets_refined/Mintaka/type/mintaka_test_Meta-Llama-3-8B-Instruct.jsonl" \
--output="results/aqe/mintaka_test_Meta-Llama-3-8B-Instruct_refined.json" 


# mintaka type_domain
python s6_aqe.py \
--benchmark="train_split" \
--train_data="datasets/d4_datasets_refined/Mintaka/type_domain/mintaka_category_A_Meta-Llama-3-8B-Instruct_somewords_token50.jsonl" \
--dev_data="" \
--test_data="datasets/d4_datasets_refined/Mintaka/type_domain/mintaka_category_B_Meta-Llama-3-8B-Instruct_somewords_token50.jsonl" \
--output="results/aqe/mintaka_test_Meta-Llama-3-8B-Instruct_OOD.json" 



# pararel ID
python s6_aqe.py \
--benchmark="train_split" \
--train_data="datasets/d4_datasets_refined/ParaRel/original/pararel_training_data_Meta-Llama-3-8B-Instruct.jsonl" \
--dev_data="" \
--test_data="datasets/d4_datasets_refined/ParaRel/original/pararel_ID_Meta-Llama-3-8B-Instruct.jsonl" \
--output="results/aqe/pararel_ID_Meta-Llama-3-8B-Instruct.json" 

# pararel OOD
python s6_aqe.py \
--benchmark="split" \
--train_data="datasets/d4_datasets_refined/ParaRel/original/pararel_training_data_Meta-Llama-3-8B-Instruct.jsonl" \
--dev_data="datasets/d4_datasets_refined/ParaRel/original/pararel_ID_Meta-Llama-3-8B-Instruct.jsonl" \
--test_data="datasets/d4_datasets_refined/ParaRel/original/pararel_OOD_Meta-Llama-3-8B-Instruct.jsonl" \
--output="results/aqe/pararel_OOD_Meta-Llama-3-8B-Instruct.jsonl" 



# pararel ID
python s6_aqe.py \
--benchmark="train_split" \
--train_data="/convei_nas2/ybseo/idk_study/make_idk_tf/results/pararel/training_data_Meta-Llama-3-8B-Instruct_somewords_token50.jsonl" \
--dev_data="" \
--test_data="/convei_nas2/ybseo/idk_study/make_idk_tf/results/pararel/ID_test_Meta-Llama-3-8B-Instruct_somewords_token50.jsonl" \
--output="results/aqe/pararel_ID_Meta-Llama-3-8B-Instruct_token50.json" 

# pararel OOD
python s6_aqe.py \
--benchmark="split" \
--train_data="/convei_nas2/ybseo/idk_study/make_idk_tf/results/pararel/training_data_Meta-Llama-3-8B-Instruct_somewords_token50.jsonl" \
--dev_data="/convei_nas2/ybseo/idk_study/make_idk_tf/results/pararel/ID_test_Meta-Llama-3-8B-Instruct_somewords_token50.jsonl" \
--test_data="/convei_nas2/ybseo/idk_study/make_idk_tf/results/pararel/OOD_test_Meta-Llama-3-8B-Instruct_somewords_token50.jsonl" \
--output="results/aqe/pararel_OOD_Meta-Llama-3-8B-Instruct_token50.json" 



# hotpotqa  original
python s6_aqe.py \
--benchmark="train_split" \
--train_data="datasets/d4_datasets_refined/HotpotQA/hotpot_10k_Meta-Llama-3-8B-Instruct.jsonl" \
--dev_data="" \
--test_data="datasets/d4_datasets_refined/HotpotQA/hotpot_test_Meta-Llama-3-8B-Instruct.jsonl" \
--output="results/aqe/hotpot_test_Meta-Llama-3-8B-Instruct_original.json" 


# hotpotqa  refined 
python s6_aqe.py \
--benchmark="train_split" \
--train_data="datasets/d4_datasets_refined/HotpotQA/type/hotpot_10k_Meta-Llama-3-8B-Instruct.jsonl" \
--dev_data="" \
--test_data="datasets/d4_datasets_refined/HotpotQA/type/hotpot_test_Meta-Llama-3-8B-Instruct.jsonl" \
--output="results/aqe/hotpot_test_Meta-Llama-3-8B-Instruct_refined.json" 




# halueval
python s6_aqe.py \
--benchmark="asone" \
--train_data="datasets/d4_datasets_refined/HaluEval/halueval_Meta-Llama-3-8B-Instruct.jsonl" \
--dev_data="" \
--test_data="" \
--output="results/aqe/halueval_Meta-Llama-3-8B-Instruct.json" 



# explain
python s6_aqe.py \
--benchmark="split" \
--train_data="datasets/d4_datasets_refined/Explain/original/explain_train_Meta-Llama-3-8B-Instruct.jsonl" \
--dev_data="datasets/d4_datasets_refined/Explain/original/explain_dev_Meta-Llama-3-8B-Instruct.jsonl" \
--test_data="datasets/d4_datasets_refined/Explain/original/explain_test_Meta-Llama-3-8B-Instruct.jsonl" \
--output="results/aqe/explain_test_Meta-Llama-3-8B-Instruct.json" 



# explain  domain
python s6_aqe.py \
--benchmark="train_split" \
--train_data="datasets/d4_datasets_refined/Explain/domain/explain_category_A_Meta-Llama-3-8B-Instruct.jsonl" \
--dev_data="" \
--test_data="datasets/d4_datasets_refined/Explain/domain/explain_category_B_Meta-Llama-3-8B-Instruct.jsonl" \
--output="results/aqe/explain_OOD_Meta-Llama-3-8B-Instruct.json" 