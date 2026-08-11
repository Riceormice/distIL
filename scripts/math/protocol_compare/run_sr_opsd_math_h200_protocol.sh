#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${1:?Usage: $0 table_aligned|github_original}"
ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-${ROOT}/code/distIL-sr-opsd-renyi}"
ENV_DIR="${ENV_DIR:-/media/vlm-ckp-fileset/ylong/sdpo/envs/verl-vllm010-h200-v2}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"
MODEL_PATH="${MODEL_PATH:-/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sr_opsd_math_protocol_compare_20260811}"

case "${PROFILE}" in
  table_aligned)
    RUN_NAME="sr-opsd-8b-seed0-table-aligned-rho0.95-refw0.9-sync0-lr5e-6-tok16384-steps100"
    TRAIN_BATCH_SIZE=8
    PPO_MINI_BATCH_SIZE=8
    ROLLOUT_N=1
    WARMUP_STEPS=0
    LR_SCHEDULER_TYPE=linear
    WEIGHT_DECAY=0
    GRAD_CLIP=0.1
    TEACHER_UPDATE_RATE=0.05
    MAX_RESPONSE_LENGTH=16384
    ROLLOUT_TEMPERATURE=0.7
    ROLLOUT_TOP_K=20
    DISTILLATION_ADD_TAIL=False
    DISTILLATION_IS_CLIP=null
    TOKEN_LOSS_CLIP=0.05
    LORA_RANK=0
    ;;
  github_original)
    RUN_NAME="sr-opsd-8b-seed0-github-original-rho0.95-refw0.9-sync0-lr5e-6-tok8192-steps100"
    TRAIN_BATCH_SIZE=32
    PPO_MINI_BATCH_SIZE=32
    ROLLOUT_N=8
    WARMUP_STEPS=10
    LR_SCHEDULER_TYPE=constant
    WEIGHT_DECAY=0.01
    GRAD_CLIP=1.0
    TEACHER_UPDATE_RATE=0.01
    MAX_RESPONSE_LENGTH=8192
    ROLLOUT_TEMPERATURE=0.8
    ROLLOUT_TOP_K=-1
    DISTILLATION_ADD_TAIL=True
    DISTILLATION_IS_CLIP=2.0
    TOKEN_LOSS_CLIP=null
    LORA_RANK=64
    ;;
  *)
    echo "PROFILE must be table_aligned or github_original, got ${PROFILE}" >&2
    exit 2
    ;;
esac

unset PYTHONHOME
unset CONDA_PREFIX
export PATH="${ENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}/SDPO:${REPO}"
export PYTHON_BIN
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/media/vlm-ckp-fileset/ylong/sdpo/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/media/vlm-ckp-fileset/ylong/sdpo/cache/datasets}"

test -x "${PYTHON_BIN}"
test -f "${REPO}/scripts/math/run_sr_opsd_verl_math_pipeline.sh"
test -f "${REPO}/SDPO/datasets/math_probs/train.json"
test -f "${REPO}/SDPO/datasets/math_probs/test.json"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  test -f "${MODEL_PATH}/config.json"
  gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  [[ "${gpu_count}" == "8" ]] || { echo "ERROR: expected 8 GPUs, found ${gpu_count}" >&2; exit 2; }
  if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Ev 'H200|H20Z' >/dev/null; then
    echo "ERROR: this launcher requires eight H200/H20Z GPUs" >&2
    nvidia-smi --query-gpu=index,name --format=csv,noheader >&2
    exit 2
  fi
fi

export TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE ROLLOUT_N
export WARMUP_STEPS LR_SCHEDULER_TYPE WEIGHT_DECAY GRAD_CLIP TEACHER_UPDATE_RATE
export MAX_RESPONSE_LENGTH ROLLOUT_TEMPERATURE ROLLOUT_TOP_K
export DISTILLATION_ADD_TAIL DISTILLATION_IS_CLIP TOKEN_LOSS_CLIP LORA_RANK
export LORA_ALPHA=128

cd "${REPO}"
cat <<EOF
============================================================
SR-OPSD Mathematics protocol comparison
host=$(hostname)
profile=${PROFILE}
run=${RUN_NAME}
model=${MODEL_PATH}
output=${OUTPUT_ROOT}
temporary_checkpoint=100 only; deleted after successful evaluation
external_eval=AIME24,AIME25,HMMT25,AMC23,Minerva; thinking; N=64
train_batch=${TRAIN_BATCH_SIZE}, mini_batch=${PPO_MINI_BATCH_SIZE}, rollout_n=${ROLLOUT_N}
response_length=${MAX_RESPONSE_LENGTH}, temperature=${ROLLOUT_TEMPERATURE}, top_p=0.95, top_k=${ROLLOUT_TOP_K}
lr=5e-6, schedule=${LR_SCHEDULER_TYPE}, warmup=${WARMUP_STEPS}
weight_decay=${WEIGHT_DECAY}, grad_clip=${GRAD_CLIP}, teacher_update=${TEACHER_UPDATE_RATE}
rho=0.95, self_reference_weight=0.9, ref_sync=0
tail=${DISTILLATION_ADD_TAIL}, distillation_is_clip=${DISTILLATION_IS_CLIP}, token_loss_clip=${TOKEN_LOSS_CLIP}
lora_rank=${LORA_RANK}
============================================================
EOF
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  nvidia-smi
fi

exec env \
  MODEL_SIZE=8b \
  MODEL_PATH="${MODEL_PATH}" \
  SEED=0 \
  TOTAL_STEPS=100 \
  SAVE_FREQ=100 \
  SELF_REFERENCE_WEIGHT=0.9 \
  REF_SYNC_STEPS=0 \
  LEARNING_RATE=5e-6 \
  DIVERGENCE_ALPHA=0.25 \
  RENYI_ORDER=0.95 \
  ENTROPY_COEFF=1e-5 \
  MAX_PROMPT_LENGTH=2048 \
  MAX_REPROMPT_LENGTH=16384 \
  ROLLOUT_TOP_P=0.95 \
  VAL_ROLLOUT_N=1 \
  VAL_TEMPERATURE=0.7 \
  VAL_TOP_P=0.95 \
  VAL_TOP_K="${ROLLOUT_TOP_K}" \
  NUM_GPUS=8 \
  DATA_DIR="${REPO}/SDPO/datasets/math_probs" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  RUN_NAME="${RUN_NAME}" \
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}" \
  EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.90}" \
  KEEP_MERGED_MODELS=0 \
  KEEP_TRAINING_CHECKPOINTS=0 \
  TRAINER_LOGGER='[console,file]' \
  PHASE=all \
  bash scripts/math/run_sr_opsd_verl_math_pipeline.sh
