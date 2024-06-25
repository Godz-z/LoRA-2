python -m torch.distributed.launch --master_port=8679 --nproc_per_node=1 \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name sst2 \
--apply_lora --apply_adalora --lora_type svd \
--target_rank 2  --lora_r 8  \
--reg_orth_coef 0.1 \
--init_warmup 6000 --final_warmup 22000 --mask_interval 100 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 16 \
--do_train --do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--learning_rate 8e-4 \
--num_train_epochs 24 \
--warmup_steps 1000 --cls_dropout 0. --weight_decay 0.01 \
--evaluation_strategy steps --eval_steps 3000 \
--save_strategy steps --save_steps 10000 \
--logging_steps 500 \
--tb_writter_loginterval 500 \
--report_to tensorboard  \
--seed 6 \
--root_output_dir ./output/debertav3-base/sst2 \
--overwrite_output_dir 
