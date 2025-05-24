# Quantifying Self-Awareness of Knowledge in Large Language Models

---

### abstract

Hallucination prediction in large language models (LLMs) is often interpreted as a sign of self-awareness. However, we argue that such performance can arise from question-side shortcuts rather than true model-side introspection. To disentangle these factors, we propose the Approximate Question-side Effect (AQE), which quantifies the contribution of question-awareness. Our analysis across multiple datasets reveals that much of the reported success stems from exploiting superficial patterns in questions. We further introduce SCAO (Semantic Compression by Answering in One word), a method that enhances the use of model-side signals. Experiments show that SCAO achieves strong and consistent performance, particularly in settings with reduced question-side cues, highlighting its effectiveness in fostering genuine self-awareness in LLMs.

# Installation

---

```coq
$ conda create --name aqe python=3.9
$ conda activate aqe
$ pip install torch==2.1.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install transformers==4.43.2 trl==0.7.4 accelerate==0.30.1 git+https://github.com/huggingface/peft.git
$ pip install datasets bitsandbytes==0.42.0 einops wandb==0.17.4
$pip install scipy==1.10.1
$pip install spacy==3.7.4
$python -m spacy download en_core_web_sm
$pip install jsonlines matplotlib addict sentencepiece
$pip install openai==0.28.1 requests numpy
```

### Must Read

The **(target)** file of the conda environment must be replaced to (**replacement**).

```coq
**(target)** [scao env dir] > site-packages > torch > nn > modules > moduel.py

**(replacement) [**project dir] > replacement > module.py
```

This replacement facilitates the injection of parameters into a model without raising an exception. Because this replacement file is specific to torch2.1.2.

---

# Data preprocess

We present all benchmark datasets preprocessed, dir ``./datasets/d1_benchmarks'': ParaRel, Mintaka, Explain, HotpotQA

Each preprocessing detail is in the paper.

# step1. Answer Generation

For answer generation, run following bash command

```jsx
bash scripts/sk1_1_answer_generation_mintaka.sh
```

s1_1_answer_generation_pararel.sh  contains following script

```jsx
python s1_answer_generation.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d1_benchmarks/Mintaka/mintaka_train.jsonl" \
--output="./datasets/d2_datasets_answered/Mintaka/mintaka_train_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"

python s1_answer_generation.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d1_benchmarks/Mintaka/mintaka_dev.jsonl" \
--output="./datasets/d2_datasets_answered/Mintaka/mintaka_dev_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"

python s1_answer_generation.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d1_benchmarks/Mintaka/mintaka_test.jsonl" \
--output="./datasets/d2_datasets_answered/Mintaka/mintaka_test_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"
```

You should input your huggingface authorization token, which is authorized to download llama3 models. Also you should input your cache directory to download it.

# step2. $k$ labeling

### 1) short-from question answering (ParaRelOOD, Mintaka, HaluEval, HotpotQA)

For $k$ labeling of each datasets, run following bash command

```bash
bash scripts/sk2_1_k_label_mintaka.sh
```

`sk2_1_k_label_mintaka.sh`  contains following script

```bash
python s2_1_k_label_short_form.py \
--test_data="./datasets/d2_datasets_answered/Mintaka/mintaka_train_Meta-Llama-3-8B-Instruct.jsonl"

...
```

### 2) long-form question answering (Explain)

For $k$ labeling of each datasets, run following bash command

```jsx
bash scripts/sk2_3_k_label_explain.sh
```

`sk2_3_k_label_explain.sh`  contains following script

```bash
python s2_2_k_label_long_form.py \
--test_file="./datasets/d2_datasets_answered/Explain/explain_train_Meta-Llama-3-8B-Instruct.jsonl" \
--api_key="your_openai_api_key"

...
```

You should input your openai api key for the ``api_key'' parameter.

And further parse the G-eval statements into k_label value, through  **`s2_2_g_eval_parse.ipynb`** file. Follow the instruction in the file.

# step4. Extracting information from LLM (confidence, hidden state)

In this step, we extract the information of LLM (confience, hidden state of the first token of the answer) and make dataset for train and evaluate the $k$-prediction module $\phi$.

For extracting information from LLM, run following bash command.

```bash
bash scripts/sk3_1_extract_information_mintaka.sh
```

`sk3_1_extract_information_mintaka.sh`  contains following scripts

```bash
python s3_extract_information.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--test_data="./datasets/d3_datasts_k_labled/ParaRel_OOD/pararel_ID_Meta-Llama-3-8B-Instruct.jsonl" \
--token="your_token" \
--cache="your_cache"

...
```

You should input your huggingface authorization token, which is authorized to download llama3 models. Also you should input your cache directory to download it.

# Step5. Evaluation.

In this step, we evaluate the method of $k$-prediction module $\phi$ on each datasets.

Evaluation file depends on the method type.

### 1) Prediction-based

These three methods ($Probe,  \ Conf+Probe$  ) can be evaluated through following bash command.

```bash
bash scripts/sk4_1_evaluation_predict.sh
```

`*sk4_1_evaluation_predict.sh*`   contains following scripts

```bash
python s5_evaluation_prediction.py \
--model="meta-llama/Meta-Llama-3-8B-Instruct" \
--benchmark="pararel" \
--method="probe_conf_scao"
```

- ``benchmark'' argument can be assigned as one of   [mintaka, pararel,   hotpotqa,  explain]
- ``method'' argument can be assigned as one of [ probe_dnn, probe_conf, probe_conf_scao ]

### 2) Threshold-based

Threshold based methods ($Conf$) can be evaluated through following bash command.

```bash
python s5_evaluation_threshold.py \
--model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--benchmark="pararel" \
--method="conf_scao"
```

- ``benchmark'' argument can be assigned as one of  [mintaka, pararel,   hotpotqa,  explain]
- ``method'' argument can be assigned as one of [ conf, conf_scao]

# Step6. AQE

You can evaluate AQE through following bash command.

```bash
python s6_aqe.py \
--benchmark="split" \
--train_data="datasets/d4_datasets_refined/Mintaka/original/mintaka_train_Meta-Llama-3-8B-Instruct.jsonl" \
--dev_data="datasets/d4_datasets_refined/Mintaka/original/mintaka_dev_Meta-Llama-3-8B-Instruct.jsonl" \
--test_data="datasets/d4_datasets_refined/Mintaka/original/mintaka_test_Meta-Llama-3-8B-Instruct.jsonl" \
--output="results/aqe/mintaka_test_Meta-Llama-3-8B-Instruct_original.json" 
```