#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SDPO_DIR="${REPO_ROOT}/SDPO"
MODEL_SIZE="${MODEL_SIZE:-8b}"

case "${MODEL_SIZE}" in
  4b)
    MODEL_PATH="${MODEL_PATH:-/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-4B-Instruct-2507}"
    ;;
  8b)
    MODEL_PATH="${MODEL_PATH:-/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-8B}"
    ;;
  *)
    echo "MODEL_SIZE must be 4b or 8b, got ${MODEL_SIZE}" >&2
    exit 2
    ;;
esac

exec env \
  PROJECT_ROOT="${SDPO_DIR}" \
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}" \
  DATA_DIR="${DATA_DIR:-${SDPO_DIR}/datasets/math_probs}" \
  MODEL_PATH="${MODEL_PATH}" \
  OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sr_opsd_math_native}" \
  RUN_NAME="${RUN_NAME:-}" \
  SEED="${SEED:-0}" \
  NUM_GPUS="${NUM_GPUS:-8}" \
  TOTAL_STEPS="${TOTAL_STEPS:-100}" \
  TEST_FREQ="${TEST_FREQ:--1}" \
  SAVE_FREQ="${SAVE_FREQ:-20}" \
  SELF_REFERENCE_WEIGHT="${SELF_REFERENCE_WEIGHT:-0.9}" \
  REF_SYNC_STEPS="${REF_SYNC_STEPS:-0}" \
  LORA_RANK="${LORA_RANK:-0}" \
  TRAINER_LOGGER="${TRAINER_LOGGER:-[console,file]}" \
  bash "${SDPO_DIR}/run_local_ours_math.sh" "${RUN_SUFFIX:-pipeline}"
