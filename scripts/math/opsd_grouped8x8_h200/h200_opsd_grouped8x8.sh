#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-${ROOT}/code/distIL-sr-opsd-renyi}"
SCRIPT_DIR="${REPO}/scripts/math/opsd_grouped8x8_h200"

export ROOT REPO
export OPSD_CODE_ROOT="${REPO}/OPSD"
export OPSD_REPO_ROOT="${REPO}"
export BASE_MODEL_DIR="/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-8B"
export MATH_TRAIN_DATA="${OPSD_CODE_ROOT}/data/math/train.jsonl"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/math_opsd_grouped8x8_eval5_n16_h200_20260820}"
export MAX_STEPS="${MAX_STEPS:-100}"
export SCHEDULER_HORIZON_STEPS="${SCHEDULER_HORIZON_STEPS:-420}"
export EVAL_FREQUENCY="${EVAL_FREQUENCY:-5}"
export VAL_N="${VAL_N:-16}"
export SEED=0
export LR=5e-6
export GRADIENT_ACCUMULATION_STEPS=8
export GROUPED_UNIQUE_PROMPTS_PER_STEP=8
export GROUPED_ROLLOUTS_PER_PROMPT=8
export ROLLOUT_TEMPERATURE=0.7
export TOP_P=0.95
export TOP_K=20
export MAX_COMPLETION_LENGTH=16384
export JSD_TOKEN_CLIP=0.06
export RUN_NAME_OVERRIDE="${RUN_NAME_OVERRIDE:-opsd-8b-seed0-grouped-q8-r8-lr5e-6-steps${MAX_STEPS}-sched${SCHEDULER_HORIZON_STEPS}-beta0-clip0.06-topk100-temp0.7-tok16384-eval${EVAL_FREQUENCY}-n${VAL_N}-h200}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29720}"

source "${REPO}/scripts/math/unified_env/activate_unified_math_env.sh" opsd

PYTHONPATH="${OPSD_CODE_ROOT}:${REPO}" "${ENV_DIR}/bin/python" \
  "${SCRIPT_DIR}/verify_grouped8x8.py" \
  --dataset "${MATH_TRAIN_DATA}" \
  --expected-dataset-size 758 \
  --world-size 8 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --rollouts-per-prompt 8 \
  --seed 0

exec bash "${REPO}/scripts/math/train_eval5_n16_h200/run_distil_method_h200.sh" opsd
