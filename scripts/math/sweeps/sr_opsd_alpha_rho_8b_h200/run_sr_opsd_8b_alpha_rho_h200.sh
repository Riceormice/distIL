#!/usr/bin/env bash
set -Eeuo pipefail

SELF_REFERENCE_WEIGHT="${1:?Usage: $0 SELF_REFERENCE_WEIGHT RENYI_ORDER}"
RENYI_ORDER="${2:?Usage: $0 SELF_REFERENCE_WEIGHT RENYI_ORDER}"

case "${SELF_REFERENCE_WEIGHT},${RENYI_ORDER}" in
  0.9,0.7|0.9,0.9|0.7,0.7|0.7,0.9|0.7,0.95) ;;
  *)
    echo "Unsupported sweep point: alpha=${SELF_REFERENCE_WEIGHT}, rho=${RENYI_ORDER}" >&2
    echo "Allowed points: (0.9,0.7), (0.9,0.9), (0.7,0.7), (0.7,0.9), (0.7,0.95)" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-${ROOT}/code/distIL-sr-opsd-renyi}"
ENV_DIR="${ENV_DIR:-/media/vlm-ckp-fileset/ylong/sdpo/envs/math-verl-current}"
MODEL_PATH="${MODEL_PATH:-/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sr_opsd_math_alpha_rho_sweep_20260819}"

export ROOT REPO ENV_DIR MODEL_PATH OUTPUT_ROOT
export MODEL_SIZE=8b
export HARDWARE=h200
export RENYI_ORDER SELF_REFERENCE_WEIGHT
export DIVERGENCE_ALPHA=0.25
export ENABLE_ADAPTIVE_GPU_KEEPALIVE="${ENABLE_ADAPTIVE_GPU_KEEPALIVE:-1}"
export KEEPALIVE_MIN_UTILIZATION="${KEEPALIVE_MIN_UTILIZATION:-38}"
export KEEPALIVE_MATRIX_SIZE="${KEEPALIVE_MATRIX_SIZE:-2048}"
export KEEPALIVE_BURST_SECONDS="${KEEPALIVE_BURST_SECONDS:-0.7}"
export KEEPALIVE_IDLE_SECONDS="${KEEPALIVE_IDLE_SECONDS:-0.8}"
export KEEPALIVE_STARTUP_DELAY_SECONDS="${KEEPALIVE_STARTUP_DELAY_SECONDS:-30}"
export KEEPALIVE_MIN_USED_MEMORY_MIB="${KEEPALIVE_MIN_USED_MEMORY_MIB:-4096}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}"
export EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.90}"
export EVAL_SUBMISSION_MODE="${EVAL_SUBMISSION_MODE:-legacy_all_prompts}"
export EVAL_PROMPT_BATCH_SIZE="${EVAL_PROMPT_BATCH_SIZE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

echo "============================================================"
echo "Qwen3-8B Math SR-OPSD alpha/rho sweep"
echo "self_reference_alpha=${SELF_REFERENCE_WEIGHT}"
echo "renyi_rho=${RENYI_ORDER}"
echo "divergence_alpha=${DIVERGENCE_ALPHA} (fixed)"
echo "training=100 steps; evaluation=every 5 steps; N=16"
echo "evaluation_submission=${EVAL_SUBMISSION_MODE}; prompt_batch_size=${EVAL_PROMPT_BATCH_SIZE}"
echo "output=${OUTPUT_ROOT}"
echo "============================================================"

exec bash "${REPO}/scripts/math/train_eval5_n16_h200/run_verl_method_h200.sh" sr_opsd
