#!/bin/bash
# Resolve OPSD directory relative to this script
OPSD_DIR="$(cd "$(dirname "$0")/../../OPSD" && pwd)"
cd "$OPSD_DIR"
# Original OPSD JSD loss — Qwen3-4B
# Usage: bash scripts/math/run_jsd_4b.sh

MODEL=${MODEL:-"Qwen/Qwen3-4B"}
DATA=${DATA:-"data/math/train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/jsd_4b"}

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes ${NUM_PROCESSES:-4} \
    --gradient_accumulation_steps 2 \
    --main_process_port 12949 \
    opsd_train.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATA} \
    --loss_mode jsd \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir ${OUTPUT_DIR} \
    --max_steps 100 \
    --max_completion_length 16384 \
    --save_steps 25 \
    --logging_steps 1 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 0.7 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --top_k_loss 100 \
    --fixed_teacher \
    --jsd_token_clip 0.05 \
    --wandb_project distil \
    --report_to wandb
