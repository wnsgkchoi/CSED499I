# Conformalized Retrieval for RAG  
This repository contains code that investigates increasing performance by formalizing RAG's retrieval process.  

This code is a modification of the rag-end2end-retriever code in huggingface's transformers library.  

To reproduce the CSED499I results, follow the steps below.  
</br>

# Note

⚠️ This project should be run with pytorch-lightning==1.3.1 which has a potential security vulnerability
</br>

# 1. Create dataset with SQuAD 1.1  
> You can skip this part because I left the dataset I used.
1. Run the 'code' through step 2 to get a file with the SQuAD dataset converted to csv. [code](data/SQuAD.ipynb)  
2. Run the code through step 3 to get a train, validation, and test set.  
3. Run the below 'code2' to extract and split data. [code2](finetune_data/squad-training-data/origin/convert.ipynb)  
</br>

# 2. Create Knowledgebase  
run this command  
```bash
python use_own_knowledge_dataset.py --csv_path finetune_data/t_squad-kb.csv --output_dir finetune_data/SQUAD-KB
```
</br>

# 3. finetuning with set1  
run 'command1'
<details>
<summary>command1</summary>
<div markdown="1">

```bash
RAY_memory_monitor_refresh_ms=0 ray start --head

python finetune_rag.py \
    --data_dir  finetune_data/squad-training-data/set1 \
    --output_dir output/model/standard/1st \
    --model_name_or_path facebook/rag-token-base \
    --model_type rag_token \
    --fp16 \
    --gpus 2  \
    --profile \
    --do_train \
    --end2end \
    --do_predict \
    --n_val -1  \
    --train_batch_size 4 \
    --eval_batch_size 1 \
    --max_source_length 128 \
    --max_target_length 25 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --lr_scheduler polynomial \
    --learning_rate 3e-05 \
    --num_train_epochs 5 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 8 \
    --distributed_retriever ray \
    --num_retrieval_workers 4  \
    --passages_path finetune_data/SQUAD-KB/my_knowledge_dataset \
    --index_path  finetune_data/SQUAD-KB/my_knowledge_dataset_hnsw_index.faiss \
    --index_name custom \
    --context_encoder_name facebook/dpr-ctx_encoder-multiset-base \
    --index_gpus 1 \
    --gpu_order [5,6,7,8,9,0,1,2,3,4] \
    --indexing_freq 500
```
</div>
</details>
</br>

# 4. find top-K  
run 'command2'

<details>
<summary>command2</summary>
<div markdown="1">

```bash
python eval_rag.py \
    --model_name_or_path output/model/standard/1st/checkpoint1251 \
    --model_type rag_token \
    --evaluation_set finetune_data/squad-training-data/set2/val.source \
    --gold_data_path finetune_data/squad-training-data/set2/val.retrieval \
    --predictions_path output/preds/1st_retrieval_preds.tsv \
    --eval_mode retrieval \
    --n_docs 1000 \
    --get_K \
    --error_rate 0.5 
```
</div>
</details>
</br>

# 5. finetuning standard and top-k model  

First, you must change line 112 in this code.  
[finetuning code](finetune_rag.py)  

When you finetune standard model('command3'), config.n_docs = 5  
When you finetune top-k model('command4'), you should set config.n_docs to K that you got with 'command2'  

<details>
<summary>command3</summary>
<div markdown="1">

```bash
python finetune_rag.py \
    --data_dir  finetune_data/squad-training-data/set2 \
    --output_dir output/model/2nd/standard \
    --model_name_or_path output/model/1st/checkpoint1251 \
    --model_type rag_token \
    --fp16 \
    --gpus 2  \
    --profile \
    --do_train \
    --end2end \
    --do_predict \
    --n_val -1  \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --max_source_length 128 \
    --max_target_length 25 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --lr_scheduler polynomial \
    --learning_rate 3e-05 \
    --num_train_epochs 3 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 8 \
    --distributed_retriever ray \
    --num_retrieval_workers 4  \
    --passages_path finetune_data/SQUAD-KB/my_knowledge_dataset \
    --index_path  finetune_data/SQUAD-KB/my_knowledge_dataset_hnsw_index.faiss \
    --index_name custom \
    --context_encoder_name facebook/dpr-ctx_encoder-multiset-base \
    --index_gpus 1 \
    --gpu_order [5,6,7,8,9,0,1,2,3,4] \
    --indexing_freq 500
```
</div>
</details>

<details>
<summary>command4</summary>
<div markdown="1">

```bash
python finetune_rag.py \
    --data_dir  finetune_data/squad-training-data/set2 \
    --output_dir output/model/2nd/top-k \
    --model_name_or_path output/model/1st/checkpoint1251 \
    --model_type rag_token \
    --fp16 \
    --gpus 2  \
    --profile \
    --do_train \
    --end2end \
    --do_predict \
    --n_val -1  \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --max_source_length 128 \
    --max_target_length 25 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --lr_scheduler polynomial \
    --learning_rate 3e-05 \
    --num_train_epochs 3 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 8 \
    --distributed_retriever ray \
    --num_retrieval_workers 4  \
    --passages_path finetune_data/SQUAD-KB/my_knowledge_dataset \
    --index_path  finetune_data/SQUAD-KB/my_knowledge_dataset_hnsw_index.faiss \
    --index_name custom \
    --context_encoder_name facebook/dpr-ctx_encoder-multiset-base \
    --index_gpus 1 \
    --gpu_order [5,6,7,8,9,0,1,2,3,4] \
    --indexing_freq 500
```
</div>
</details>
</br>

# 6. evaluation standard and top-k model  
run 'command5' to get scores of standard model.
run 'command6' to get scores of top-k model.

<details>
<summary>command5</summary>
<div markdown="1">

```bash
python eval_rag.py \
    --model_name_or_path output/model/2nd/standard/checkpoint316 \
    --model_type rag_token \
    --evaluation_set finetune_data/squad-training-data/set2/test.source \
    --gold_data_path finetune_data/squad-training-data/set2/test.target \
    --predictions_path output/preds/2st_standard_preds.tsv \
    --eval_mode e2e \
    --gold_data_mode ans \
    --n_docs 5 \
    --k 1 \
    --print_predictions \
    --recalculate
```
</div>
</details>
<details>
<summary>command6</summary>
<div markdown="1">

```bash
python eval_rag.py \
    --model_name_or_path output/model/2nd/top-k/checkpoint751 \
    --model_type rag_token \
    --evaluation_set finetune_data/squad-training-data/set2/test.source \
    --gold_data_path finetune_data/squad-training-data/set2/test.target \
    --predictions_path output/preds/2st_top-k_preds.tsv \
    --eval_mode e2e \
    --gold_data_mode ans \
    --n_docs 5 \
    --print_predictions \
    --recalculate
```
</div>
</details>
</br>

# 7. Result  
|metric|Standard|Top-K|
|-|-|-|
|f1 score|18.44|18.88|
|em score|12.40|11.20|
