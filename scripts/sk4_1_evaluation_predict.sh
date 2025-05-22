


##### mintaka
echo !!!!!mintaka original!!!!!
python s5_evaluation_prediction.py \
--benchmark='split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Mintaka/original/mintaka_train_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="datasets/d5_datasets_processed/Mintaka/original/mintaka_dev_Meta-Llama-3-8B-Instruct.pt" \
--test_data="datasets/d5_datasets_processed/Mintaka/original/mintaka_test_Meta-Llama-3-8B-Instruct.pt" \
--method='probe_conf_scao' \
--output='original/mintaka_test_Meta-Llama-3-8B-Instruct'

python s5_evaluation_prediction.py \
--benchmark='split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Mintaka/original/mintaka_train_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="datasets/d5_datasets_processed/Mintaka/original/mintaka_dev_Meta-Llama-3-8B-Instruct.pt" \
--test_data="datasets/d5_datasets_processed/Mintaka/original/mintaka_test_Meta-Llama-3-8B-Instruct.pt" \
--method='probe_conf' \
--output='original/mintaka_test_Meta-Llama-3-8B-Instruct'


python s5_evaluation_prediction.py \
--benchmark='split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Mintaka/original/mintaka_train_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="datasets/d5_datasets_processed/Mintaka/original/mintaka_dev_Meta-Llama-3-8B-Instruct.pt" \
--test_data="datasets/d5_datasets_processed/Mintaka/original/mintaka_test_Meta-Llama-3-8B-Instruct.pt" \
--method='probe_dnn' \
--output='original/mintaka_test_Meta-Llama-3-8B-Instruct'


####  type

echo !!!!!mintaka type !!!!!
python s5_evaluation_prediction.py \
--benchmark='split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Mintaka/type/mintaka_train_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="datasets/d5_datasets_processed/Mintaka/type/mintaka_dev_Meta-Llama-3-8B-Instruct.pt" \
--test_data="datasets/d5_datasets_processed/Mintaka/type/mintaka_test_Meta-Llama-3-8B-Instruct.pt" \
--method='probe_conf_scao' \
--output='type/mintaka_test_Meta-Llama-3-8B-Instruct'

python s5_evaluation_prediction.py \
--benchmark='split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Mintaka/type/mintaka_train_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="datasets/d5_datasets_processed/Mintaka/type/mintaka_dev_Meta-Llama-3-8B-Instruct.pt" \
--test_data="datasets/d5_datasets_processed/Mintaka/type/mintaka_test_Meta-Llama-3-8B-Instruct.pt" \
--method='probe_conf' \
--output='type/mintaka_test_Meta-Llama-3-8B-Instruct'


python s5_evaluation_prediction.py \
--benchmark='split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Mintaka/type/mintaka_train_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="datasets/d5_datasets_processed/Mintaka/type/mintaka_dev_Meta-Llama-3-8B-Instruct.pt" \
--test_data="datasets/d5_datasets_processed/Mintaka/type/mintaka_test_Meta-Llama-3-8B-Instruct.pt" \
--method='probe_dnn' \
--output='type/mintaka_test_Meta-Llama-3-8B-Instruct'


## domain

echo !!!!!mintaka refine !!!!!
python s5_evaluation_prediction.py \
--benchmark='train_split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Mintaka/type_domain/mintaka_category_A_Meta-Llama-3-8B-Instruct_somewords.pt" \
--dev_data="" \
--test_data="datasets/d5_datasets_processed/Mintaka/type_domain/mintaka_category_B_Meta-Llama-3-8B-Instruct_somewords.pt" \
--output='type_domain/mintaka_test_Meta-Llama-3-8B-Instruct' \
--method='probe_conf_scao'


python s5_evaluation_prediction.py \
--benchmark='train_split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Mintaka/type_domain/mintaka_category_A_Meta-Llama-3-8B-Instruct_somewords.pt" \
--dev_data="" \
--test_data="datasets/d5_datasets_processed/Mintaka/type_domain/mintaka_category_B_Meta-Llama-3-8B-Instruct_somewords.pt" \
--output='type_domain/mintaka_test_Meta-Llama-3-8B-Instruct' \
--method='probe_conf'


python s5_evaluation_prediction.py \
--benchmark='train_split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Mintaka/type_domain/mintaka_category_A_Meta-Llama-3-8B-Instruct_somewords.pt" \
--dev_data="" \
--test_data="datasets/d5_datasets_processed/Mintaka/type_domain/mintaka_category_B_Meta-Llama-3-8B-Instruct_somewords.pt" \
--output='type_domain/mintaka_test_Meta-Llama-3-8B-Instruct' \
--method='probe_dnn'



##### hotpotQA

echo !!!!!hotpotQA original !!!!!
python s5_evaluation_prediction.py \
--benchmark='train_split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/HotpotQA/hotpot_10k_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="" \
--test_data="datasets/d5_datasets_processed/HotpotQA/hotpot_test_Meta-Llama-3-8B-Instruct.pt" \
--output='original/hotpot_test_Meta-Llama-3-8B-Instruct' \
--method='probe_conf_scao'


python s5_evaluation_prediction.py \
--benchmark='train_split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/HotpotQA/hotpot_10k_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="" \
--test_data="datasets/d5_datasets_processed/HotpotQA/hotpot_test_Meta-Llama-3-8B-Instruct.pt" \
--output='original/hotpot_test_Meta-Llama-3-8B-Instruct' \
--method='probe_conf'


python s5_evaluation_prediction.py \
--benchmark='train_split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/HotpotQA/hotpot_10k_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="" \
--test_data="datasets/d5_datasets_processed/HotpotQA/hotpot_test_Meta-Llama-3-8B-Instruct.pt" \
--output='original/hotpot_test_Meta-Llama-3-8B-Instruct' \
--method='probe_dnn'


## type
echo !!!!!hotpotQA type !!!!!
python s5_evaluation_prediction.py \
--benchmark='train_split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/HotpotQA/type/hotpot_10k_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="" \
--test_data="datasets/d5_datasets_processed/HotpotQA/type/hotpot_test_Meta-Llama-3-8B-Instruct.pt" \
--output='type/hotpot_test_Meta-Llama-3-8B-Instruct' \
--method='probe_conf_scao'

python s5_evaluation_prediction.py \
--benchmark='train_split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/HotpotQA/type/hotpot_10k_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="" \
--test_data="datasets/d5_datasets_processed/HotpotQA/type/hotpot_test_Meta-Llama-3-8B-Instruct.pt" \
--output='type/hotpot_test_Meta-Llama-3-8B-Instruct' \
--method='probe_conf'


python s5_evaluation_prediction.py \
--benchmark='train_split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/HotpotQA/type/hotpot_10k_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="" \
--test_data="datasets/d5_datasets_processed/HotpotQA/type/hotpot_test_Meta-Llama-3-8B-Instruct.pt" \
--output='type/hotpot_test_Meta-Llama-3-8B-Instruct' \
--method='probe_dnn'


#### explain


echo !!!!!explain original!!!!!
python s5_evaluation_prediction.py \
--benchmark='split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Explain/explain_train_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="datasets/d5_datasets_processed/Explain/explain_dev_Meta-Llama-3-8B-Instruct.pt" \
--test_data="datasets/d5_datasets_processed/Explain/explain_test_Meta-Llama-3-8B-Instruct.pt" \
--output='original/explain_test_Meta-Llama-3-8B-Instruct.pt' \
--method='probe_conf_scao'


python s5_evaluation_prediction.py \
--benchmark='split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Explain/explain_train_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="datasets/d5_datasets_processed/Explain/explain_dev_Meta-Llama-3-8B-Instruct.pt" \
--test_data="datasets/d5_datasets_processed/Explain/explain_test_Meta-Llama-3-8B-Instruct.pt" \
--output='original/explain_test_Meta-Llama-3-8B-Instruct.pt' \
--method='probe_conf'


python s5_evaluation_prediction.py \
--benchmark='split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Explain/explain_train_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="datasets/d5_datasets_processed/Explain/explain_dev_Meta-Llama-3-8B-Instruct.pt" \
--test_data="datasets/d5_datasets_processed/Explain/explain_test_Meta-Llama-3-8B-Instruct.pt" \
--output='original/explain_test_Meta-Llama-3-8B-Instruct.pt' \
--method='probe_dnn'



####  domain

echo !!!!!explain refine !!!!!
python s5_evaluation_prediction.py \
--benchmark='train_split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Explain/explain_category_A_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="" \
--test_data="datasets/d5_datasets_processed/Explain/explain_category_B_Meta-Llama-3-8B-Instruct.pt" \
--output='domain/explain_test_Meta-Llama-3-8B-Instruct.pt' \
--method='probe_conf_scao'


python s5_evaluation_prediction.py \
--benchmark='train_split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Explain/explain_category_A_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="" \
--test_data="datasets/d5_datasets_processed/Explain/explain_category_B_Meta-Llama-3-8B-Instruct.pt" \
--output='domain/explain_test_Meta-Llama-3-8B-Instruct.pt' \
--method='probe_conf' 


python s5_evaluation_prediction.py \
--benchmark='train_split' \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--train_data="datasets/d5_datasets_processed/Explain/explain_category_A_Meta-Llama-3-8B-Instruct.pt" \
--dev_data="" \
--test_data="datasets/d5_datasets_processed/Explain/explain_category_B_Meta-Llama-3-8B-Instruct.pt" \
--output='domain/explain_test_Meta-Llama-3-8B-Instruct.pt' \
--method='probe_dnn' 

