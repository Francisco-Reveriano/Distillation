#!/bin/bash

uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-7B-Instruct"
lr=1e-5
epochs=10
weight_decay=1e-4
micro_batch_size=1
gradient_accumulation_steps=1
gpu_count=$(nvidia-smi -L | wc -l)
push_to_hub=false

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    Train/sft.py \
    --block_size=32768 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="simplescaling/s1K_tokenized" \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="Train/fsdp_config_qwen.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="ckpts/s1-${uid}" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True

# If you want to enable gradient checkpointing later, you can add it in a separate invocation or uncomment the following lines:
# --gradient_checkpointing=True \
# --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'
