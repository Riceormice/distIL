#!/bin/bash
# Resolve OPSD directory relative to this script
OPSD_DIR="$(cd "$(dirname "$0")/../../OPSD" && pwd)"
cd "$OPSD_DIR"
# SFT baseline — Qwen3-4B
# Usage: bash scripts/math/run_sft_4b.sh

MODEL=${MODEL:-"Qwen/Qwen3-4B"}
DATA=${DATA:-"data/math/train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/sft_4b"}

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes ${NUM_PROCESSES:-4} \
    --gradient_accumulation_steps 4 \
    --main_process_port 12952 \
    sft_train.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATA} \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --output_dir ${OUTPUT_DIR} \
    --max_steps 100 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_length 16000 \
    --logging_steps 5 \
    --save_steps 100 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16
