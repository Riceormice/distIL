#!/usr/bin/env bash
set -Eeuo pipefail

SELF_REFERENCE_WEIGHT="${1:?Usage: $0 SELF_REFERENCE_WEIGHT RENYI_ORDER}"
RENYI_ORDER="${2:?Usage: $0 SELF_REFERENCE_WEIGHT RENYI_ORDER}"

case "${SELF_REFERENCE_WEIGHT},${RENYI_ORDER}" in
  0.9,0.7|0.9,0.9|0.7,0.7|0.7,0.9|0.7,0.95) ;;
  *)
    echo "ERROR: unsupported 4B alpha/rho point: ${SELF_REFERENCE_WEIGHT},${RENYI_ORDER}" >&2
    echo "Allowed points: (0.9,0.7), (0.9,0.9), (0.7,0.7), (0.7,0.9), (0.7,0.95)" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sr_opsd_math_4b_alpha_rho_sweep_eval5_n16_a800_20260829}"

export REPO OUTPUT_ROOT
export MODEL_SIZE=4b
export HARDWARE=a800
export DIVERGENCE_ALPHA=0.25
export RENYI_ORDER SELF_REFERENCE_WEIGHT
export EVAL_SUBMISSION_MODE=legacy_all_prompts
export EVAL_PROMPT_BATCH_SIZE=0
export ENABLE_ADAPTIVE_GPU_KEEPALIVE="${ENABLE_ADAPTIVE_GPU_KEEPALIVE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

echo "============================================================"
echo "Qwen3-4B Math SR-OPSD alpha/rho sweep"
echo "self_reference_alpha=${SELF_REFERENCE_WEIGHT}"
echo "renyi_rho=${RENYI_ORDER}"
echo "divergence_alpha=${DIVERGENCE_ALPHA} (fixed implementation parameter)"
echo "hardware=8 x A800"
echo "training=100 steps; evaluation=every 5 steps; N=16"
echo "evaluation=legacy all-prompts; max_new_tokens=16384"
echo "output=${OUTPUT_ROOT}"
echo "============================================================"

exec bash "${REPO}/scripts/math/train_eval5_n16_a800_4b/run_verl_method_a800_4b.sh" sr_opsd
