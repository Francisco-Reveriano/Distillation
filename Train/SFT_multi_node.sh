uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-7B-Instruct" # meta-llama/Llama-3.1-70B-Instruct
lr=1e-5
min_lr=0
epochs=10
micro_batch_size=1 # If 2 nodes with 8 gpus each, batch_size will be 16
push_to_hub=true
gradient_accumulation_steps=1
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)

torchrun \
--nproc-per-node ${gpu_count} \
Train/sft.py \
--per_device_train_batch_size=${micro_batch_size} \
--per_device_eval_batch_size=${micro_batch_size} \
--gradient_accumulation_steps=${gradient_accumulation_steps} \
--train_file_path="simplescaling/s1K_tokenized" \
--block_size=32768 \
--model_name=${base_model} \
--warmup_ratio=0.05 \
--fsdp="full_shard auto_wrap" \
--fsdp_config="Train/fsdp_config_qwen.json" \
--bf16=True \
--eval_strategy="steps" \
--eval_steps=50 \
--logging_steps=1 \
--save_steps=100 \
--lr_scheduler_type cosine \
--learning_rate ${lr} \
--weight_decay 1e-4 \
--adam_beta1 0.9 \
--adam_beta2 0.95 \
--output_dir="ckpts/s1_${uid}" \
--num_train_epochs ${epochs} \
--save_only_model=True \
--gradient_checkpointing=True