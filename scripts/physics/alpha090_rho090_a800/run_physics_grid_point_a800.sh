#!/usr/bin/env bash
set -euo pipefail

# Run one point from the Qwen3-8B Physics rho x self-reference grid.
# The implementation selector alpha remains fixed at 0.25 (Forward Renyi),
# while SELF_REFERENCE_COEFFICIENT maps to the paper's self-reference alpha.

METHOD=${1:?"method is required"}
RHO_ARG=${2:-}

ROOT=${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}
PROJECT_ROOT=${PROJECT_ROOT:-"${ROOT}/code/SDPO-main-latest-provided"}
PYTHON_ENV=${PYTHON_ENV:-/media/damoxing/che-liu-fileset/ylong/sdpo/envs/math-verl-current}
PYTHON_BIN=${PYTHON_BIN:-"${PYTHON_ENV}/bin/python"}
OUTPUT_ROOT=${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sdpo_physics_rho_selfref_grid_eval5_nockpt}
MODEL_PATH=${MODEL_PATH:-/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-8B}
TRAIN_FILE=${TRAIN_FILE:-"${PROJECT_ROOT}/datasets/sciknoweval/physics/train.parquet"}
VAL_FILE=${VAL_FILE:-"${PROJECT_ROOT}/datasets/sciknoweval/physics/test.parquet"}
CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-"${PROJECT_ROOT}/verl/utils/reward_score/feedback/__init__.py"}

: "${SEED:?SEED must be set to 0}"
if [[ "${SEED}" != "0" ]]; then
  echo "ERROR: this controlled grid is fixed to SEED=0; got ${SEED}" >&2
  exit 2
fi

N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-420}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
ROLLOUT_N=${ROLLOUT_N:-8}
LR=${LR:-1e-5}
WARMUP_STEPS=${WARMUP_STEPS:-10}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-100}
TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE:-0.05}
ENTROPY_COEFF=${ENTROPY_COEFF:-1e-5}
: "${SELF_REFERENCE_COEFFICIENT:?Set SELF_REFERENCE_COEFFICIENT to 0.5, 0.7, or 0.9}"
case "${SELF_REFERENCE_COEFFICIENT}" in
  0.5|0.7|0.9) ;;
  *)
    echo "ERROR: expected SELF_REFERENCE_COEFFICIENT=0.5, 0.7, or 0.9; got ${SELF_REFERENCE_COEFFICIENT}" >&2
    exit 2
    ;;
esac
REFERENCE_WEIGHT=${SELF_REFERENCE_COEFFICIENT}
REF_SYNC_STEPS=${REF_SYNC_STEPS:-0}
IS_CLIP=${IS_CLIP:-2.0}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
VAL_N=${VAL_N:-16}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}
TEST_FREQ=${TEST_FREQ:-5}
SAVE_FREQ=${SAVE_FREQ:--1}
EXPERIMENT_SUFFIX=${EXPERIMENT_SUFFIX:-rho-selfref-grid-v1}
PROJECT_NAME=${PROJECT_NAME:-SR-OPSD-Physics-Rho-SelfRef-Grid}
GROUP_NAME=${GROUP_NAME:-physics_rho_selfref_grid_seed${SEED}}

if [[ "${REF_SYNC_STEPS}" != "0" ]]; then
  echo "ERROR: this controlled replication requires REF_SYNC_STEPS=0" >&2
  exit 2
fi

if [[ "${METHOD}" != "sr_opsd_ref" ]]; then
  echo "ERROR: this grid only supports method=sr_opsd_ref; got ${METHOD}" >&2
  exit 2
fi

METHOD_LABEL=sr_opsd_forward_renyi
OBJECTIVE_SELECTOR_ALPHA=0.25
RAW_RHO=${RHO_ARG:?"rho is required for sr_opsd_ref"}
case "${RAW_RHO}" in
  0.5|0.7|0.9|0.95) ;;
  *)
    echo "ERROR: expected SR-OPSD rho 0.5, 0.7, 0.9, or 0.95; got ${RAW_RHO}" >&2
    exit 2
    ;;
esac
RHO_LABEL=${RAW_RHO}
USE_REFERENCE=True
REG_LEVEL=${REFERENCE_WEIGHT}

# Keep the path component below 255 bytes; the full model path is still logged
# in the resolved Hydra command.
MODEL_NAME=$(basename "${MODEL_PATH}")
EXP_NAME="physics-${METHOD_LABEL}-refTrue-selectorAlpha${OBJECTIVE_SELECTOR_ALPHA}-rho${RHO_LABEL}-selfref${REG_LEVEL}-sync0-entropy${ENTROPY_COEFF}-ema${TEACHER_UPDATE_RATE}-topk${DISTILLATION_TOPK}-tailTrue-fullLogitTrue-isclip${IS_CLIP}-steps${TOTAL_TRAINING_STEPS}-trainbs${TRAIN_BATCH_SIZE}-mbs${PPO_MINI_BATCH_SIZE}-rolloutn${ROLLOUT_N}-lr${LR}-warmup${WARMUP_STEPS}-seed${SEED}-model${MODEL_NAME}-${EXPERIMENT_SUFFIX}"
RUN_DIR="${OUTPUT_ROOT}/runs/${EXP_NAME}"
RUN_LOG_DIR="${OUTPUT_ROOT}/logs/${EXP_NAME}"

for required_path in \
  "${PROJECT_ROOT}" "${PYTHON_BIN}" "${MODEL_PATH}" \
  "${TRAIN_FILE}" "${VAL_FILE}" "${CUSTOM_REWARD_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: required path does not exist: ${required_path}" >&2
    exit 3
  fi
done

mkdir -p "${RUN_DIR}" "${RUN_LOG_DIR}"
if [[ -f "${RUN_DIR}/TRAINING_COMPLETE" ]]; then
  echo "SKIP: completed run already exists: ${RUN_DIR}"
  exit 0
fi

export PATH="${PYTHON_ENV}/bin:/usr/local/cuda/bin:/usr/bin:/bin:${PATH:-}"
export VIRTUAL_ENV="${PYTHON_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}"
export LD_LIBRARY_PATH="${PYTHON_ENV}/lib:${PYTHON_ENV}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export SETUPTOOLS_USE_DISTUTILS=stdlib
export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export RAY_DEDUP_LOGS=0
export USER=${USER:-$(whoami)}
export TASK=physics
export N_GPUS_PER_NODE
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_USE_MODELSCOPE=true
export VLLM_USE_V1=1
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VERL_FILE_LOGGER_PATH="${RUN_LOG_DIR}/metrics.jsonl"
export TENSORBOARD_DIR="${RUN_LOG_DIR}/tensorboard"
export SWANLAB_MODE=disabled
export SDPO_SWANLAB_MODE=disabled
export SWANLAB_DISABLED=1
unset PYTHONHOME PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
unset SWANLAB_API_KEY SWANLAB_WORKSPACE SWANLAB_PROJECT
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_MODE

ARGS=(
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${VAL_FILE}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.seed=${SEED}"
  "actor_rollout_ref.actor.data_loader_seed=${SEED}"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.actor.optim.lr=${LR}"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=${WARMUP_STEPS}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.calculate_entropy=True"
  "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}"
  "actor_rollout_ref.actor.self_distillation.full_logit_distillation=True"
  "actor_rollout_ref.actor.self_distillation.distillation_topk=${DISTILLATION_TOPK}"
  "actor_rollout_ref.actor.self_distillation.distillation_add_tail=True"
  "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True"
  "actor_rollout_ref.actor.self_distillation.alpha=${OBJECTIVE_SELECTOR_ALPHA}"
  "actor_rollout_ref.actor.self_distillation.rho=${RAW_RHO}"
  "actor_rollout_ref.actor.self_distillation.renyi_regularization=${USE_REFERENCE}"
  "actor_rollout_ref.actor.self_distillation.renyi_regularization_level=${REG_LEVEL}"
  "actor_rollout_ref.actor.self_distillation.renyi_ref_sync_steps=0"
  "actor_rollout_ref.actor.self_distillation.teacher_update_rate=${TEACHER_UPDATE_RATE}"
  "actor_rollout_ref.actor.self_distillation.is_clip=${IS_CLIP}"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
  "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}"
  "actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K}"
  "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}"
  "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}"
  "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}"
  "actor_rollout_ref.rollout.enforce_eager=True"
  "algorithm.rollout_correction.rollout_is=token"
  "custom_reward_function.path=${CUSTOM_REWARD_PATH}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.group_name=${GROUP_NAME}"
  "trainer.experiment_name=${EXP_NAME}"
  "trainer.default_local_dir=${RUN_DIR}"
  "trainer.logger=[console,file,tensorboard]"
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
  "trainer.nnodes=${NNODES}"
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.max_actor_ckpt_to_keep=null"
  "trainer.max_critic_ckpt_to_keep=null"
  "trainer.val_before_train=False"
  "trainer.resume_mode=disable"
  "trainer.log_val_generations=16"
  "trainer.validation_data_dir=${RUN_LOG_DIR}/validation"
  "trainer.rollout_data_dir=${RUN_LOG_DIR}/rollouts"
)

if [[ -n "${RAY_TEMP_DIR:-}" ]]; then
  mkdir -p "${RAY_TEMP_DIR}"
  ARGS+=(
    "+ray_kwargs.ray_init._temp_dir=${RAY_TEMP_DIR}"
    "+ray_kwargs.ray_init.include_dashboard=False"
  )
fi

printf '%s\n' \
  "seed=${SEED}" \
  "method=${METHOD}" \
  "effective_rho=${RHO_LABEL}" \
  "reference=${USE_REFERENCE}" \
  "implementation_objective_selector_alpha=${OBJECTIVE_SELECTOR_ALPHA}" \
  "self_reference_coefficient=${REG_LEVEL}" \
  "experiment=${EXP_NAME}" \
  "run_dir=${RUN_DIR}" \
  "log_dir=${RUN_LOG_DIR}"
printf '%q ' "${PYTHON_BIN}" -m verl.trainer.main_ppo --config-name sdpo "${ARGS[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${PYTHON_BIN}" -m verl.trainer.main_ppo --config-name sdpo "${ARGS[@]}" \
  2>&1 | tee -a "${RUN_LOG_DIR}/train.log"
touch "${RUN_DIR}/TRAINING_COMPLETE"
