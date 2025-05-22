
echo !!!!!mintaka original!!!!!
python s5_evaluation_threshold.py \
--benchmark='split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Mintaka/original/mintaka_train_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="datasets/d5_datasets_processed/Mintaka/original/mintaka_dev_Meta-Llama-3-8B-Instruct.pt" \
--test_data="datasets/d5_datasets_processed/Mintaka/original/mintaka_test_Meta-Llama-3-8B-Instruct.pt" \
--output='original/mintaka_test_Meta-Llama-3-8B-Instruct'


echo !!!!!mintaka type !!!!!
python s5_evaluation_threshold.py \
--benchmark='split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Mintaka/type/mintaka_train_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="datasets/d5_datasets_processed/Mintaka/type/mintaka_dev_Meta-Llama-3-8B-Instruct.pt" \
--test_data="datasets/d5_datasets_processed/Mintaka/type/mintaka_test_Meta-Llama-3-8B-Instruct.pt" \
--output='type/mintaka_test_Meta-Llama-3-8B-Instruct'


echo !!!!!mintaka domain !!!!!
python s5_evaluation_threshold.py \
--benchmark='train_split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Mintaka/type_domain/mintaka_category_A_Meta-Llama-3-8B-Instruct_somewords_token50.pt" \
--dev_data="" \
--test_data="datasets/d5_datasets_processed/Mintaka/type_domain/mintaka_category_B_Meta-Llama-3-8B-Instruct_somewords_token50.pt" \
--output='type_domain/mintaka_test_Meta-Llama-3-8B-Instruct.pt'



#### hotpotQA

echo !!!!!hotpotQA original !!!!!
python s5_evaluation_threshold.py \
--benchmark='train_split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/HotpotQA/hotpot_10k_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="" \
--test_data="datasets/d5_datasets_processed/HotpotQA/hotpot_test_Meta-Llama-3-8B-Instruct.pt" \
--output='original/hotpot_test_Meta-Llama-3-8B-Instruct' \

echo !!!!!hotpotQA type !!!!!
python s5_evaluation_threshold.py \
--benchmark='train_split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/HotpotQA/type/hotpot_10k_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="" \
--test_data="datasets/d5_datasets_processed/HotpotQA/type/hotpot_test_Meta-Llama-3-8B-Instruct.pt" \
--output='type/hotpot_test_Meta-Llama-3-8B-Instruct'


### explain
echo !!!!!explain original !!!!!
python s5_evaluation_threshold.py \
--benchmark='split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Explain/explain_train_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="datasets/d5_datasets_processed/Explain/explain_dev_Meta-Llama-3-8B-Instruct.pt" \
--test_data="datasets/d5_datasets_processed/Explain/explain_test_Meta-Llama-3-8B-Instruct.pt" \
--output='original/explain_test_Meta-Llama-3-8B-Instruct.pt'


echo !!!!!explain domain !!!!!
python s5_evaluation_threshold.py \
--benchmark='train_split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Explain/explain_category_A_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="" \
--test_data="datasets/d5_datasets_processed/Explain/explain_category_B_Meta-Llama-3-8B-Instruct.pt" \
--output='domain/explain_test_Meta-Llama-3-8B-Instruct.pt' 


## pararel
echo !!!!!pararel original!!!!!
python s5_evaluation_threshold.py \
--benchmark='train_split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d4_datasets_processed/ParaRel/pararel_training_data_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="" \
--test_data="datasets/d4_datasets_processed/ParaRel/pararel_ID_Meta-Llama-3-8B-Instruct.pt" \
--output='original/pararel_ID_Meta-Llama-3-8B-Instruct'


echo !!!!!pararel domain !!!!!
python s5_evaluation_threshold.py \
--benchmark='split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d4_datasets_processed/ParaRel/pararel_training_data_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="datasets/d4_datasets_processed/ParaRel/pararel_ID_Meta-Llama-3-8B-Instruct.pt" \
--test_data="datasets/d4_datasets_processed/ParaRel/pararel_domain_Meta-Llama-3-8B-Instruct.pt" \
--output='domain/pararel_ID_Meta-Llama-3-8B-Instruct'