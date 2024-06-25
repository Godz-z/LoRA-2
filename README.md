# LoRA$^2$ : Multi-Scale Low-Rank Approximations for Fine-Tuning Large Language Models

Fine-tuning large language models (LLMs) with high parameter efficiency for downstream tasks has become a new paradigm. Low-Rank Adaptation (LoRA) significantly reduces the number of trainable parameters for fine-tuning. Although it has demonstrated commendable performance, updating parameters within a single scale may not be the optimal choice for complex downstream tasks. In this paper, we extend the LoRA to multiple scales, dubbed as LoRA$^2$. We first combine orthogonal projection theory to train a set of LoRAs in two mutually orthogonal planes. Then, we improve the importance score algorithm, which reduce parameter sensitivity score calculations by approximately 98.5\%. By pruning singular values with lower importance scores, thereby enhancing adaptability to various downstream tasks. Extensive experiments are conducted on two widely used pre-trained models to validate the effectiveness of LoRA$^2$. Results show that it significantly reduces the number of trainable parameters to just 0.72\% compared to full fine-tuning, while still delivering highly impressive performance. Even when the parameters are further reduced to 0.17M, it still achieves comparable results to the baseline with 8 times more parameters.


# Adapating DeBERTaV3 with LoRA$^2$

The folder contains the implementation of LoRA$^2$ in DeBERTaV3 using the updated package of `loralib`, which contains the implementation of LoRA$^2$. LoRA$^2$ is present the following paper: [LoRA$^2$ : Multi-Scale Low-Rank Approximations for Fine-Tuning Large Language Models]. 


## Setup Environment

### Create and activate the conda env
```bash
conda create -n NLU python=3.7
conda activate NLU 
```

### Install Pytorch
```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### Install the pre-requisites
Install dependencies: 
```bash
pip install -r requirements.txt
```

Install `transformers`: (here we fork NLU examples from [microsoft/LoRA](https://github.com/microsoft/LoRA/tree/main/examples/NLU) and build our examples based on their `transformers` version, which is `v4.4.2`.)
```bash
pip install -e . 
```

Install the updated `loralib`:
```bash
pip install -e ../loralib/
```


## Adapt DeBERTaV3 on GLUE benchmark

### The example to reproduce the MNLI results

```bash
python -m torch.distributed.launch --master_port=8679 --nproc_per_node=1 \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name mnli \
--apply_adalora --apply_lora --lora_type svd \
--target_rank 3  --lora_r 8  \
--reg_orth_coef 0.1 \
--init_warmup 8000 --final_warmup 50000 --mask_interval 100 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 16 \
--do_train --do_eval \
--max_seq_length 256 \
--per_device_train_batch_size 32 --learning_rate 5e-4 --num_train_epochs 7 \
--warmup_steps 1000 \
--cls_dropout 0.15 --weight_decay 0 \
--evaluation_strategy steps --eval_steps 3000 \
--save_strategy steps --save_steps 30000 \
--logging_steps 500 \
--seed 6 \
--root_output_dir ./output/glue/mnli \
--overwrite_output_dir


#### Hyperparameter Setup

+ `apply_lora`: Apply LoRA to the target model. 
+ `lora_type`: Config the low-rank parameterization, `frd` for low-rank decomposition and `svd` for SVD decomposition. Use `svd` for AdaLoRA and `frd` for LoRA. 
+ `apply_adalora`: Further apply AdaLoRA for the model that have been modified by LoRA. 
+ `lora_module`: The types of modules updated by LoRA. 
+ `lora_r`: The initial rank of each incremental matrix. 
+ `target_rank`: The average target rank of final incremental matrices, i.e. the average number of singular values per matrix. 
+ `init_warmup`: The steps of initial warmup for budget scheduler.
+ `final_warmup`: The steps of final warmup for budget scheduler. 
+ `mask_interval`: The time internval between two budget allocations.
+ `beta1` and `beta2`: The coefficient of exponentional moving average when updating importance scores. 
+ `reg_orth_coef`: The weight of orthongonal regularization. 


