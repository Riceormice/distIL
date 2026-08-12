#!/usr/bin/env bash
set -Eeuo pipefail

METHOD="${1:?Usage: $0 grpo|sdpo|sr_opsd}"
case "${METHOD}" in
  grpo|sdpo|sr_opsd) ;;
  *) echo "ERROR: METHOD must be grpo, sdpo, or sr_opsd" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

exec env \
  REPO="${REPO}" \
  MODEL_SIZE=4b \
  HARDWARE=a800 \
  MODEL_PATH="${MODEL_PATH:-/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-4B-Instruct-2507}" \
  ENV_DIR="${ENV_DIR:-/media/vlm-ckp-fileset/ylong/sdpo/envs/verl-vllm010-h200-v2}" \
  OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/math_4b_train_eval5_n16_a800_20260812}" \
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}" \
  EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.90}" \
  bash "${REPO}/scripts/math/train_eval5_n16_h200/run_verl_method_h200.sh" "${METHOD}"
