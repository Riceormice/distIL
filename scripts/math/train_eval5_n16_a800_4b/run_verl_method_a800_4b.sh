#!/usr/bin/env bash
set -Eeuo pipefail

METHOD="${1:?Usage: $0 grpo|sdpo|sr_opsd}"
case "${METHOD}" in
  grpo|sdpo|sr_opsd) ;;
  *) echo "ERROR: METHOD must be grpo, sdpo, or sr_opsd" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  for candidate in \
    /media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-4B \
    /media/vlm-ckp-fileset/ylong/models/Qwen3-4B \
    /media/damoxing/che-liu-fileset/ylong/sdpo/models/Qwen3-4B
  do
    if [[ -s "${candidate}/config.json" ]]; then
      MODEL_PATH="${candidate}"
      break
    fi
  done
fi

if [[ -z "${MODEL_PATH:-}" ]]; then
  while IFS= read -r config_path; do
    MODEL_PATH="${config_path%/config.json}"
    break
  done < <(
    find /media/vlm-ckp-fileset/ylong /media/damoxing/che-liu-fileset/ylong \
      -maxdepth 7 -type f -path '*/Qwen3-4B/config.json' \
      -print 2>/dev/null
  )
fi

if [[ -z "${MODEL_PATH:-}" || ! -s "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: Qwen3-4B was not found." >&2
  echo "Set MODEL_PATH to its exact local directory before launching." >&2
  exit 2
fi

echo "Resolved Qwen3-4B: ${MODEL_PATH}"

exec env \
  REPO="${REPO}" \
  MODEL_SIZE=4b \
  HARDWARE=a800 \
  MODEL_PATH="${MODEL_PATH}" \
  ENV_DIR="${ENV_DIR:-/media/damoxing/che-liu-fileset/ylong/sdpo/envs/math-verl-current}" \
  OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/math_4b_train_eval5_n16_a800_20260812}" \
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}" \
  EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.90}" \
  bash "${REPO}/scripts/math/train_eval5_n16_h200/run_verl_method_h200.sh" "${METHOD}"
