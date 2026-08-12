#!/usr/bin/env bash
set -Eeuo pipefail

METHOD="${1:?Usage: $0 grpo|sdpo|sr_opsd [suffix]}"
case "${METHOD}" in
  grpo|sdpo|sr_opsd) ;;
  *) echo "ERROR: METHOD must be grpo, sdpo, or sr_opsd" >&2; exit 2 ;;
esac

SUFFIX="${2:-native-verl-math}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/math_probs}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/math_native_verl}"

SEED="${SEED:-0}"
NUM_GPUS="${NUM_GPUS:-${N_GPUS_PER_NODE:-8}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TOTAL_STEPS="${TOTAL_STEPS:-420}"
STOP_AFTER_STEP="${STOP_AFTER_STEP:-0}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
TEST_FREQ="${TEST_FREQ:--1}"
SAVE_FREQ="${SAVE_FREQ:-5}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-[model,optimizer,extra]}"
CHECKPOINT_LOAD_CONTENTS="${CHECKPOINT_LOAD_CONTENTS:-${CHECKPOINT_SAVE_CONTENTS}}"

# Shared optimization and generation protocol. Method-specific objectives are
# selected below; these values intentionally stay identical across all lanes.
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
GRAD_CLIP="${GRAD_CLIP:-0.1}"
ENTROPY_COEFF="${ENTROPY_COEFF:-1e-5}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
MAX_REPROMPT_LENGTH="${MAX_REPROMPT_LENGTH:-16384}"
# The teacher consumes the reprompt and the sampled response together. The
# context limit must therefore cover that longer path, not only prompt+response.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_REPROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-20}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.7}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"
VAL_TOP_K="${VAL_TOP_K:-20}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$}"

# Shared self-distillation settings used by SDPO and SR-OPSD.
DISTILLATION_TOPK="${DISTILLATION_TOPK:-100}"
DISTILLATION_ADD_TAIL="${DISTILLATION_ADD_TAIL:-False}"
DISTILLATION_IS_CLIP="${DISTILLATION_IS_CLIP:-null}"
TOKEN_LOSS_CLIP="${TOKEN_LOSS_CLIP:-0.05}"
TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"

case "${METHOD}" in
  grpo)
    CONFIG_NAME=baseline_grpo
    METHOD_LABEL=GRPO
    METHOD_SLUG=grpo
    SAVE_TEACHER_CHECKPOINT=False
    METHOD_SUMMARY="GRPO; epsilon=0.2; group-normalized advantages"
    ;;
  sdpo)
    CONFIG_NAME=sdpo
    METHOD_LABEL="SDPO (RKL)"
    METHOD_SLUG=sdpo
    SAVE_TEACHER_CHECKPOINT=True
    DIVERGENCE_ALPHA=1.0
    RENYI_REGULARIZATION=False
    RENYI_ORDER=0.95
    SELF_REFERENCE_WEIGHT=0.0
    REF_SYNC_STEPS=0
    METHOD_SUMMARY="Reverse KL self-distillation; EMA teacher; no frozen-reference anchoring"
    ;;
  sr_opsd)
    CONFIG_NAME=sdpo
    METHOD_LABEL=SR-OPSD
    METHOD_SLUG=sr-opsd
    SAVE_TEACHER_CHECKPOINT=True
    DIVERGENCE_ALPHA="${DIVERGENCE_ALPHA:-0.25}"
    RENYI_REGULARIZATION=True
    RENYI_ORDER="${RENYI_ORDER:-0.95}"
    SELF_REFERENCE_WEIGHT="${SELF_REFERENCE_WEIGHT:-0.9}"
    REF_SYNC_STEPS="${REF_SYNC_STEPS:-0}"
    METHOD_SUMMARY="Forward Renyi; rho=${RENYI_ORDER}; self-reference=${SELF_REFERENCE_WEIGHT}; frozen reference"
    ;;
esac

"${PYTHON_BIN}" - "${METHOD}" "${ROLLOUT_N}" "${STOP_AFTER_STEP}" "${TOTAL_STEPS}" \
  "${MAX_MODEL_LEN}" "${MAX_PROMPT_LENGTH}" "${MAX_RESPONSE_LENGTH}" "${MAX_REPROMPT_LENGTH}" <<'PY'
import sys

method = sys.argv[1]
rollout_n = int(sys.argv[2])
stop_after = int(sys.argv[3])
total_steps = int(sys.argv[4])
max_model_len = int(sys.argv[5])
max_prompt_len = int(sys.argv[6])
max_response_len = int(sys.argv[7])
max_reprompt_len = int(sys.argv[8])

if rollout_n < 2:
    raise SystemExit("ROLLOUT_N must be at least 2: GRPO needs a non-degenerate group, and all lanes share this value")
if stop_after < 0 or stop_after > total_steps:
    raise SystemExit(f"STOP_AFTER_STEP must be in [0, {total_steps}], got {stop_after}")
if max_model_len < max_prompt_len + max_response_len:
    raise SystemExit("MAX_MODEL_LEN must cover MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH")
if method != "grpo" and max_model_len < max_reprompt_len + max_response_len:
    raise SystemExit("MAX_MODEL_LEN must cover MAX_REPROMPT_LENGTH + MAX_RESPONSE_LENGTH for self-distillation")
PY

if [[ "${METHOD}" == "sr_opsd" ]]; then
  "${PYTHON_BIN}" - "${DIVERGENCE_ALPHA}" "${RENYI_ORDER}" "${SELF_REFERENCE_WEIGHT}" "${REF_SYNC_STEPS}" <<'PY'
import math
import sys

alpha = float(sys.argv[1])
rho = float(sys.argv[2])
weight = float(sys.argv[3])
sync_steps = int(sys.argv[4])
if not math.isclose(alpha, 0.25):
    raise SystemExit(f"SR-OPSD Forward Renyi requires DIVERGENCE_ALPHA=0.25, got {alpha}")
if not math.isfinite(rho) or rho <= 0 or math.isclose(rho, 1.0):
    raise SystemExit(f"RENYI_ORDER must be positive and different from 1, got {rho}")
if not 0.0 <= weight <= 1.0:
    raise SystemExit(f"SELF_REFERENCE_WEIGHT must be in [0, 1], got {weight}")
if sync_steps < 0:
    raise SystemExit(f"REF_SYNC_STEPS must be non-negative, got {sync_steps}")
PY
fi

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
RUN_NAME="${RUN_NAME:-${METHOD_SLUG}-${MODEL_NAME}-seed${SEED}-lr${LEARNING_RATE}-trainbs${TRAIN_BATCH_SIZE}-mbs${PPO_MINI_BATCH_SIZE}-rolloutn${ROLLOUT_N}-steps${TOTAL_STEPS}-${SUFFIX}}"
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
  "max_model_len=${MAX_MODEL_LEN}"
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
  "actor_rollout_ref.actor.clip_ratio=0.2"
  "actor_rollout_ref.actor.clip_ratio_low=0.2"
  "actor_rollout_ref.actor.clip_ratio_high=0.2"
  "actor_rollout_ref.actor.checkpoint.save_contents=${CHECKPOINT_SAVE_CONTENTS}"
  "actor_rollout_ref.actor.checkpoint.load_contents=${CHECKPOINT_LOAD_CONTENTS}"

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
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN}"

  "algorithm.adv_estimator=grpo"
  "algorithm.rollout_correction.rollout_is=token"
  "algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_IS_THRESHOLD}"
  "custom_reward_function.path=${PROJECT_ROOT}/verl/utils/reward_score/feedback/__init__.py"
  "reward_model.use_reward_loop=False"

  "trainer.project_name=${PROJECT_NAME:-Math-Native-VERL}"
  "trainer.group_name=${GROUP_NAME:-native-verl-math}"
  "trainer.experiment_name=${RUN_NAME}"
  "trainer.logger=${TRAINER_LOGGER:-[console,file]}"
  "trainer.n_gpus_per_node=${NUM_GPUS}"
  "trainer.nnodes=1"
  "trainer.total_training_steps=${TOTAL_STEPS}"
  "trainer.stop_after_step=${STOP_AFTER_STEP}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}"
  "trainer.val_before_train=False"
  "trainer.resume_mode=auto"
  "trainer.default_local_dir=${RUN_DIR}"
)

if [[ "${METHOD}" == "grpo" ]]; then
  ARGS+=(
    "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla"
    "algorithm.norm_adv_by_std_in_grpo=True"
  )
else
  ARGS+=(
    "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo"
    "algorithm.norm_adv_by_std_in_grpo=False"
    "actor_rollout_ref.actor.self_distillation.full_logit_distillation=True"
    "actor_rollout_ref.actor.self_distillation.distillation_topk=${DISTILLATION_TOPK}"
    "actor_rollout_ref.actor.self_distillation.distillation_add_tail=${DISTILLATION_ADD_TAIL}"
    "actor_rollout_ref.actor.self_distillation.is_clip=${DISTILLATION_IS_CLIP}"
    "actor_rollout_ref.actor.self_distillation.token_loss_clip=${TOKEN_LOSS_CLIP}"
    "actor_rollout_ref.actor.self_distillation.success_reward_threshold=0.5"
    "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True"
    "actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=True"
    "actor_rollout_ref.actor.self_distillation.include_environment_feedback=True"
    "actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution=True"
    "actor_rollout_ref.actor.self_distillation.alpha=${DIVERGENCE_ALPHA}"
    "actor_rollout_ref.actor.self_distillation.rho=${RENYI_ORDER}"
    "actor_rollout_ref.actor.self_distillation.renyi_regularization=${RENYI_REGULARIZATION}"
    "actor_rollout_ref.actor.self_distillation.renyi_regularization_level=${SELF_REFERENCE_WEIGHT}"
    "actor_rollout_ref.actor.self_distillation.renyi_ref_sync_steps=${REF_SYNC_STEPS}"
    "actor_rollout_ref.actor.self_distillation.teacher_regularization=ema"
    "actor_rollout_ref.actor.self_distillation.teacher_update_rate=${TEACHER_UPDATE_RATE}"
    "actor_rollout_ref.actor.self_distillation.save_teacher_checkpoint=${SAVE_TEACHER_CHECKPOINT}"
    "actor_rollout_ref.actor.self_distillation.max_reprompt_len=${MAX_REPROMPT_LENGTH}"
  )
fi

cat <<EOF
============================================================
Native SDPO/VERL mathematics training
method=${METHOD_LABEL}
objective=${METHOD_SUMMARY}
run=${RUN_NAME}
model=${MODEL_PATH}
data=${DATA_DIR}
output=${RUN_DIR}
gpus=${NUM_GPUS}
steps=${TOTAL_STEPS}, stop_after=${STOP_AFTER_STEP}, save=${SAVE_FREQ}
train_batch=${TRAIN_BATCH_SIZE}, mini_batch=${PPO_MINI_BATCH_SIZE}, rollouts=${ROLLOUT_N}
max_prompt=${MAX_PROMPT_LENGTH}, max_response=${MAX_RESPONSE_LENGTH}
temperature=${ROLLOUT_TEMPERATURE}, top_p=${ROLLOUT_TOP_P}, top_k=${ROLLOUT_TOP_K}
lr=${LEARNING_RATE}, schedule=${LR_SCHEDULER_TYPE}, warmup=${WARMUP_STEPS}
weight_decay=${WEIGHT_DECAY}, grad_clip=${GRAD_CLIP}, entropy=${ENTROPY_COEFF}
topk=${DISTILLATION_TOPK}, tail=${DISTILLATION_ADD_TAIL}, token_clip=${TOKEN_LOSS_CLIP}
============================================================
EOF

cd "${PROJECT_ROOT}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN command:'
  printf ' %q' "${PYTHON_BIN}" -m verl.trainer.main_ppo --config-name "${CONFIG_NAME}" "${ARGS[@]}"
  printf '\n'
  exit 0
fi
exec "${PYTHON_BIN}" -m verl.trainer.main_ppo --config-name "${CONFIG_NAME}" "${ARGS[@]}"
