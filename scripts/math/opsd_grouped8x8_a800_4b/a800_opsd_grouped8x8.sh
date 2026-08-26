#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-${ROOT}/code/distIL-sr-opsd-renyi}"
VERIFY_SCRIPT="${REPO}/scripts/math/opsd_grouped8x8_h200/verify_grouped8x8.py"

export ROOT REPO
export MODEL_SIZE=4b
export HARDWARE=a800
export OPSD_CODE_ROOT="${REPO}/OPSD"
export OPSD_REPO_ROOT="${REPO}"
export MATH_TRAIN_DATA="${OPSD_CODE_ROOT}/data/math/train.jsonl"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/math_4b_opsd_grouped8x8_eval5_n16_a800_20260827}"
export MAX_STEPS="${MAX_STEPS:-100}"
export SCHEDULER_HORIZON_STEPS="${SCHEDULER_HORIZON_STEPS:-420}"
export EVAL_FREQUENCY="${EVAL_FREQUENCY:-5}"
export VAL_N="${VAL_N:-16}"
export EVAL_SUBMISSION_MODE="${EVAL_SUBMISSION_MODE:-legacy_all_prompts}"
export EVAL_PROMPT_BATCH_SIZE="${EVAL_PROMPT_BATCH_SIZE:-0}"
export SEED=0
export LR=5e-6
export GRADIENT_ACCUMULATION_STEPS=8
export GROUPED_UNIQUE_PROMPTS_PER_STEP=8
export GROUPED_ROLLOUTS_PER_PROMPT=8
export ROLLOUT_TEMPERATURE=0.7
export TOP_P=0.95
export TOP_K=20
export MAX_COMPLETION_LENGTH=16384
export JSD_TOKEN_CLIP=0.05
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
export EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.90}"
export RUN_NAME_OVERRIDE="${RUN_NAME_OVERRIDE:-opsd-4b-seed0-grouped-q8-r8-lr5e-6-steps${MAX_STEPS}-sched${SCHEDULER_HORIZON_STEPS}-beta0-clip0.05-topk100-temp0.7-tok16384-eval${EVAL_FREQUENCY}-n${VAL_N}-a800}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29730}"

source "${REPO}/scripts/math/unified_env/activate_unified_math_env.sh" opsd

PYTHONPATH="${OPSD_CODE_ROOT}:${REPO}" "${ENV_DIR}/bin/python" \
  "${VERIFY_SCRIPT}" \
  --dataset "${MATH_TRAIN_DATA}" \
  --expected-dataset-size 758 \
  --world-size 8 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --rollouts-per-prompt 8 \
  --seed 0

exec bash "${REPO}/scripts/math/train_eval5_n16_h200/run_distil_method_h200.sh" opsd
