#!/usr/bin/env bash
set -Eeuo pipefail

# Native SDPO/VERL SR-OPSD training for the mathematics dataset supplied on
# the math-train branch. All paths and experiment parameters can be overridden
# through environment variables.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/math_probs}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sr_opsd_math_native}"

SEED="${SEED:-0}"
NUM_GPUS="${NUM_GPUS:-${N_GPUS_PER_NODE:-8}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
TEST_FREQ="${TEST_FREQ:-100}"
SAVE_FREQ="${SAVE_FREQ:-100}"

LEARNING_RATE="${LEARNING_RATE:-5e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
GRAD_CLIP="${GRAD_CLIP:-0.1}"
ENTROPY_COEFF="${ENTROPY_COEFF:-1e-5}"
TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"

# The native SDPO implementation uses alpha=0.25 to select Forward Renyi.
DIVERGENCE_ALPHA="${DIVERGENCE_ALPHA:-0.25}"
RENYI_ORDER="${RENYI_ORDER:-0.95}"
SELF_REFERENCE_WEIGHT="${SELF_REFERENCE_WEIGHT:-0.9}"
REF_SYNC_STEPS="${REF_SYNC_STEPS:-0}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
MAX_REPROMPT_LENGTH="${MAX_REPROMPT_LENGTH:-16384}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-20}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.7}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"
VAL_TOP_K="${VAL_TOP_K:-20}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"

DISTILLATION_TOPK="${DISTILLATION_TOPK:-100}"
DISTILLATION_ADD_TAIL="${DISTILLATION_ADD_TAIL:-False}"
DISTILLATION_IS_CLIP="${DISTILLATION_IS_CLIP:-null}"
TOKEN_LOSS_CLIP="${TOKEN_LOSS_CLIP:-0.05}"

LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$}"

PROJECT_NAME="${PROJECT_NAME:-SR-OPSD-Math}"
GROUP_NAME="${GROUP_NAME:-native-sdpo-math}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[console,file]}"
RUN_SUFFIX="${1:-native-sdpo}"

if [[ "${DIVERGENCE_ALPHA}" != "0.25" ]]; then
  echo "ERROR: native Forward Renyi requires DIVERGENCE_ALPHA=0.25" >&2
  exit 2
fi

"${PYTHON_BIN}" - "${RENYI_ORDER}" "${SELF_REFERENCE_WEIGHT}" "${REF_SYNC_STEPS}" <<'PY'
import math
import sys

rho = float(sys.argv[1])
weight = float(sys.argv[2])
sync_steps = int(sys.argv[3])
if not math.isfinite(rho) or rho <= 0 or math.isclose(rho, 1.0):
    raise SystemExit(f"RENYI_ORDER must be positive and different from 1, got {rho}")
if not 0.0 <= weight <= 1.0:
    raise SystemExit(f"SELF_REFERENCE_WEIGHT must be in [0, 1], got {weight}")
if sync_steps < 0:
    raise SystemExit(f"REF_SYNC_STEPS must be non-negative, got {sync_steps}")
PY

TRAIN_JSON="${DATA_DIR}/train.json"
TEST_JSON="${DATA_DIR}/test.json"
TRAIN_PARQUET="${DATA_DIR}/train.parquet"
TEST_PARQUET="${DATA_DIR}/test.parquet"

test -x "${PYTHON_BIN}"
test -f "${TRAIN_JSON}"
test -f "${TEST_JSON}"
if [[ "${MODEL_PATH}" == /* ]]; then
  test -f "${MODEL_PATH}/config.json"
fi

mkdir -p "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/logs"

if [[ "${DRY_RUN:-0}" != "1" ]] && \
   [[ ! -s "${TRAIN_PARQUET}" || ! -s "${TEST_PARQUET}" || "${TRAIN_JSON}" -nt "${TRAIN_PARQUET}" || "${TEST_JSON}" -nt "${TEST_PARQUET}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT}" PYTHON_BIN="${PYTHON_BIN}" DATA_DIR="${DATA_DIR}" \
    bash "${PROJECT_ROOT}/prepare_math_data.sh"
fi

MODEL_NAME="$(basename "${MODEL_PATH}")"
RUN_NAME="${RUN_NAME:-sr-opsd-${MODEL_NAME}-seed${SEED}-alpha${DIVERGENCE_ALPHA}-rho${RENYI_ORDER}-refw${SELF_REFERENCE_WEIGHT}-sync${REF_SYNC_STEPS}-lr${LEARNING_RATE}-steps${TOTAL_STEPS}-${RUN_SUFFIX}}"
RUN_DIR="${OUTPUT_ROOT}/checkpoints/${RUN_NAME}"
LOG_DIR="${OUTPUT_ROOT}/logs/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export USER="${USER:-root}"
export TASK="datasets/math_probs"
export EXPERIMENT="${RUN_NAME}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH:-${LOG_DIR}/metrics.jsonl}"

ARGS=(
  "data.train_files=${TRAIN_PARQUET}"
  "data.val_files=${TEST_PARQUET}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.seed=${SEED}"
  "data.shuffle=True"
  'data.apply_chat_template_kwargs={enable_thinking: false}'

  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.lora_rank=${LORA_RANK}"
  "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}"
  "actor_rollout_ref.model.target_modules='${LORA_TARGET_MODULES}'"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"

  "actor_rollout_ref.actor.data_loader_seed=${SEED}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.ppo_epochs=1"
  "actor_rollout_ref.actor.loss_agg_mode=token-mean"
  "actor_rollout_ref.actor.calculate_entropy=True"
  "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}"
  "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
  "actor_rollout_ref.actor.optim.lr_scheduler_type=${LR_SCHEDULER_TYPE}"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=${WARMUP_STEPS}"
  "actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY}"
  "actor_rollout_ref.actor.grad_clip=${GRAD_CLIP}"
  "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo"

  "actor_rollout_ref.actor.self_distillation.full_logit_distillation=True"
  "actor_rollout_ref.actor.self_distillation.distillation_topk=${DISTILLATION_TOPK}"
  "actor_rollout_ref.actor.self_distillation.distillation_add_tail=${DISTILLATION_ADD_TAIL}"
  "actor_rollout_ref.actor.self_distillation.is_clip=${DISTILLATION_IS_CLIP}"
  "actor_rollout_ref.actor.self_distillation.token_loss_clip=${TOKEN_LOSS_CLIP}"
  "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True"
  "actor_rollout_ref.actor.self_distillation.alpha=${DIVERGENCE_ALPHA}"
  "actor_rollout_ref.actor.self_distillation.rho=${RENYI_ORDER}"
  "actor_rollout_ref.actor.self_distillation.renyi_regularization=True"
  "actor_rollout_ref.actor.self_distillation.renyi_regularization_level=${SELF_REFERENCE_WEIGHT}"
  "actor_rollout_ref.actor.self_distillation.renyi_ref_sync_steps=${REF_SYNC_STEPS}"
  "actor_rollout_ref.actor.self_distillation.teacher_regularization=ema"
  "actor_rollout_ref.actor.self_distillation.teacher_update_rate=${TEACHER_UPDATE_RATE}"
  "actor_rollout_ref.actor.self_distillation.max_reprompt_len=${MAX_REPROMPT_LENGTH}"

  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
  "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}"
  "actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K}"
  "actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N}"
  "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}"
  "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}"
  "actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.rollout.enforce_eager=True"

  "algorithm.rollout_correction.rollout_is=token"
  "custom_reward_function.path=${PROJECT_ROOT}/verl/utils/reward_score/feedback/__init__.py"
  "reward_model.use_reward_loop=False"

  "trainer.project_name=${PROJECT_NAME}"
  "trainer.group_name=${GROUP_NAME}"
  "trainer.experiment_name=${RUN_NAME}"
  "trainer.logger=${TRAINER_LOGGER}"
  "trainer.n_gpus_per_node=${NUM_GPUS}"
  "trainer.nnodes=1"
  "trainer.total_training_steps=${TOTAL_STEPS}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.seed=${SEED}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.max_actor_ckpt_to_keep=1"
  "trainer.val_before_train=False"
  "trainer.resume_mode=auto"
  "trainer.default_local_dir=${RUN_DIR}"
)

cat <<EOF
============================================================
Native SDPO/VERL SR-OPSD mathematics training
run=${RUN_NAME}
model=${MODEL_PATH}
data=${DATA_DIR}
output=${RUN_DIR}
gpus=${NUM_GPUS}
steps=${TOTAL_STEPS}, epochs=${TOTAL_EPOCHS}, eval=${TEST_FREQ}, save=${SAVE_FREQ}
alpha=${DIVERGENCE_ALPHA}, rho=${RENYI_ORDER}
self_reference_weight=${SELF_REFERENCE_WEIGHT}, ref_sync=${REF_SYNC_STEPS}
teacher_ema=${TEACHER_UPDATE_RATE}, lr=${LEARNING_RATE}, warmup=${WARMUP_STEPS}
schedule=${LR_SCHEDULER_TYPE}, weight_decay=${WEIGHT_DECAY}, grad_clip=${GRAD_CLIP}
topk=${DISTILLATION_TOPK}, tail=${DISTILLATION_ADD_TAIL}, is_clip=${DISTILLATION_IS_CLIP}, token_clip=${TOKEN_LOSS_CLIP}
============================================================
EOF

cd "${PROJECT_ROOT}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN command:'
  printf ' %q' "${PYTHON_BIN}" -m verl.trainer.main_ppo --config-name sdpo "${ARGS[@]}"
  printf '\n'
  exit 0
fi
exec "${PYTHON_BIN}" -m verl.trainer.main_ppo --config-name sdpo "${ARGS[@]}"
