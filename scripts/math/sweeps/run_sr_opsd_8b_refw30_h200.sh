#!/usr/bin/env bash
set -Eeuo pipefail

SELF_REFERENCE_WEIGHT="${1:?Usage: $0 SELF_REFERENCE_WEIGHT}"
ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-${ROOT}/code/distIL-sr-opsd-renyi}"
ENV_DIR="${ENV_DIR:-/media/vlm-ckp-fileset/ylong/sdpo/envs/verl-vllm010-h200-v2}"
MODEL_PATH="${MODEL_PATH:-/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sr_opsd_math_refw_sweep30}"
RUN_NAME="sr-opsd-8b-seed0-rho0.95-refw${SELF_REFERENCE_WEIGHT}-sync0-lr5e-6-tok16384-steps30-eval5"

unset PYTHONHOME
unset PYTHONPATH
unset CONDA_PREFIX

export PATH="${ENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}/SDPO"
export PYTHON_BIN="${ENV_DIR}/bin/python"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/media/vlm-ckp-fileset/ylong/sdpo/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/media/vlm-ckp-fileset/ylong/sdpo/cache/datasets}"

test -x "${PYTHON_BIN}"
test -f "${MODEL_PATH}/config.json"
test -f "${REPO}/scripts/math/run_sr_opsd_verl_math_pipeline.sh"

cd "${REPO}"
echo "host=$(hostname)"
echo "run=${RUN_NAME}"
echo "self_reference_weight=${SELF_REFERENCE_WEIGHT}"
nvidia-smi

exec env \
  MODEL_SIZE=8b \
  MODEL_PATH="${MODEL_PATH}" \
  SELF_REFERENCE_WEIGHT="${SELF_REFERENCE_WEIGHT}" \
  RUN_NAME="${RUN_NAME}" \
  NUM_GPUS=8 \
  TOTAL_STEPS=30 \
  SAVE_FREQ=5 \
  DATA_DIR="${OUTPUT_ROOT}/data/refw${SELF_REFERENCE_WEIGHT}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}" \
  EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.90}" \
  KEEP_MERGED_MODELS=0 \
  PHASE=all \
  bash scripts/math/run_sr_opsd_verl_math_pipeline.sh
