#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
MODEL_SIZE="${MODEL_SIZE:-8b}"
HARDWARE="${HARDWARE:-h200}"

case "${MODEL_SIZE}:${HARDWARE}" in
  8b:h200)
    OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/math_grpo_8b_opsd_trl_aligned_eval5_n16_h200_20260827}"
    ;;
  4b:a800)
    OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/math_grpo_4b_opsd_trl_aligned_eval5_n16_a800_20260827}"
    ;;
  *)
    echo "ERROR: supported pairs are MODEL_SIZE=8b/HARDWARE=h200 and MODEL_SIZE=4b/HARDWARE=a800" >&2
    exit 2
    ;;
esac

# Use the checked-out implementation, not a potentially stale baseline clone.
OPSD_CODE_ROOT="${OPSD_CODE_ROOT:-${REPO}/OPSD}"
OPSD_REPO_ROOT="${OPSD_REPO_ROOT:-${REPO}}"
export ROOT REPO MODEL_SIZE HARDWARE OUTPUT_ROOT OPSD_CODE_ROOT OPSD_REPO_ROOT

case "${MODEL_SIZE}:${HARDWARE}" in
  8b:h200) source "${REPO}/scripts/math/train_eval5_n16_h200/common_distil_h200.sh" ;;
  4b:a800) source "${REPO}/scripts/math/train_eval5_n16_a800_4b/common_distil_a800.sh" ;;
esac
source "${REPO}/scripts/math/lock_protocol.sh"

MAX_STEPS="${MAX_STEPS:-100}"
SCHEDULER_HORIZON_STEPS="${SCHEDULER_HORIZON_STEPS:-420}"
EVAL_FREQUENCY="${EVAL_FREQUENCY:-5}"
VAL_N="${VAL_N:-16}"
SEED="${SEED:-0}"
LR="${LR:-5e-6}"
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
UNIQUE_PROMPTS_PER_STEP=8
NUM_GENERATIONS=8
TRAJECTORIES_PER_STEP=64
NUM_ITERATIONS=1
GRPO_EPSILON=0.2
ENTROPY_COEFFICIENT=0
LOSS_TYPE=dapo
SCALE_REWARDS=group
IMPORTANCE_SAMPLING_LEVEL=token
VLLM_IS_MODE=token_mask
VLLM_IS_CAP=3.0
MAX_PROMPT_LENGTH=2048
MAX_COMPLETION_LENGTH=16384
ROLLOUT_TEMPERATURE=0.7
ROLLOUT_TOP_P=0.95
ROLLOUT_TOP_K=20
LORA_DROPOUT=0.0
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29670}"
EVAL_SUBMISSION_MODE="${EVAL_SUBMISSION_MODE:-legacy_all_prompts}"

case "${EVAL_SUBMISSION_MODE}" in
  legacy_all_prompts) EVAL_PROMPT_BATCH_SIZE=0 ;;
  chunked) EVAL_PROMPT_BATCH_SIZE="${EVAL_PROMPT_BATCH_SIZE:-8}" ;;
  *) echo "ERROR: EVAL_SUBMISSION_MODE must be legacy_all_prompts or chunked" >&2; exit 2 ;;
esac

case "${MODEL_SIZE}" in
  8b)
    EVAL_TEMPERATURE=1.0
    EVAL_TOP_P=1.0
    EVAL_TOP_K=-1
    EVAL_MAX_NEW_TOKENS=16384
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.42}"
    ;;
  4b)
    EVAL_TEMPERATURE=0.7
    EVAL_TOP_P=0.95
    EVAL_TOP_K=20
    EVAL_MAX_NEW_TOKENS=16384
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
    ;;
esac

(( MAX_STEPS > 0 )) || { echo "ERROR: MAX_STEPS must be positive" >&2; exit 2; }
(( EVAL_FREQUENCY > 0 )) || { echo "ERROR: EVAL_FREQUENCY must be positive" >&2; exit 2; }
(( MAX_STEPS % EVAL_FREQUENCY == 0 )) || {
  echo "ERROR: MAX_STEPS must be divisible by EVAL_FREQUENCY" >&2
  exit 2
}
(( SCHEDULER_HORIZON_STEPS >= MAX_STEPS )) || {
  echo "ERROR: scheduler horizon must be at least the physical stop step" >&2
  exit 2
}

SAVE_STEPS="$(seq -s, "${EVAL_FREQUENCY}" "${EVAL_FREQUENCY}" "${MAX_STEPS}")"
RUN_NAME="${RUN_NAME_OVERRIDE:-grpo-${MODEL_SIZE}-seed${SEED}-opsd-trl-q8-r8-lr${LR}-eps${GRPO_EPSILON}-lora64a128-temp${ROLLOUT_TEMPERATURE}-tok${MAX_COMPLETION_LENGTH}-steps${MAX_STEPS}-sched${SCHEDULER_HORIZON_STEPS}-eval${EVAL_FREQUENCY}-n${VAL_N}-${HARDWARE}}"
RUN_ROOT="${OUTPUT_ROOT}/grpo/${RUN_NAME}"
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"
TRAIN_OUTPUT_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"
RESULT_ROOT="${RUN_ROOT}/evaluations"
LOG_ROOT="${RUN_ROOT}/logs"
STATE_ROOT="${RUN_ROOT}/state"
METRICS_JSONL="${RUN_ROOT}/training_metrics.jsonl"
KEEPALIVE_SCRIPT="${REPO}/scripts/math/adaptive_gpu_keepalive.py"
KEEPALIVE_PIDS=()
KEEPALIVE_STOP_FILE=""

mkdir -p "${TRAIN_OUTPUT_DIR}" "${RESULT_ROOT}" "${LOG_ROOT}" "${STATE_ROOT}"
exec 9>"${STATE_ROOT}/pipeline.lock"
flock -n 9 || { echo "ERROR: GRPO pipeline is already running: ${RUN_ROOT}" >&2; exit 3; }

LAUNCH_LOG="${LOG_ROOT}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

report_error() {
  local status=$?
  local line="${BASH_LINENO[0]:-unknown}"
  local command="${BASH_COMMAND:-unknown}"
  trap - ERR
  echo "ERROR: OPSD/TRL GRPO pipeline failed: status=${status} line=${line} command=${command}" >&2
  echo "log=${LAUNCH_LOG}" >&2
  exit "${status}"
}
trap report_error ERR

cleanup_pipeline() {
  if [[ -n "${KEEPALIVE_STOP_FILE}" ]]; then
    touch "${KEEPALIVE_STOP_FILE}" 2>/dev/null || true
  fi
  local pid
  for pid in "${KEEPALIVE_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  [[ -z "${KEEPALIVE_STOP_FILE}" ]] || rm -f -- "${KEEPALIVE_STOP_FILE}"
}
trap cleanup_pipeline EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export SDPO_METRICS_JSONL="${METRICS_JSONL}"
export WANDB_MODE=disabled
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT

for required in \
  "${OPSD_CODE_ROOT}/grpo_train.py" \
  "${OPSD_CODE_ROOT}/accelerate.yaml" \
  "${KEEPALIVE_SCRIPT}"
do
  [[ -f "${required}" ]] || { echo "ERROR: missing required file: ${required}" >&2; exit 2; }
done
for control in selected_checkpoint_steps stop_after_step auto_resume save_final_model load_math_dataset JsonlMetricsCallback; do
  grep -q "${control}" "${OPSD_CODE_ROOT}/grpo_train.py" || {
    echo "ERROR: GRPO trainer lacks required control ${control}: ${OPSD_CODE_ROOT}/grpo_train.py" >&2
    exit 2
  }
done

TRAIN_DATA_SHA256="$(sha256sum "${MATH_TRAIN_DATA}" | awk '{print $1}')"
MODEL_CONFIG_SHA256="$(sha256sum "${BASE_MODEL_DIR}/config.json" | awk '{print $1}')"
TOKENIZER_CONFIG_SHA256="$(sha256sum "${BASE_MODEL_DIR}/tokenizer_config.json" | awk '{print $1}')"
TRAINER_SHA256="$(sha256sum "${OPSD_CODE_ROOT}/grpo_train.py" | awk '{print $1}')"
CODE_COMMIT="$(git -C "${REPO}" rev-parse HEAD 2>/dev/null || printf 'unknown')"

PROTOCOL_CANDIDATE="${STATE_ROOT}/protocol.env.candidate.$$"
cat >"${PROTOCOL_CANDIDATE}" <<EOF
method=grpo
framework=OPSD-TRL-Accelerate-DeepSpeed
model_size=${MODEL_SIZE}
hardware=${HARDWARE}
model_path=${BASE_MODEL_DIR}
model_config_sha256=${MODEL_CONFIG_SHA256}
tokenizer_config_sha256=${TOKENIZER_CONFIG_SHA256}
training_dataset=${MATH_TRAIN_DATA}
training_dataset_sha256=${TRAIN_DATA_SHA256}
trainer_sha256=${TRAINER_SHA256}
evaluation_data_root=${MATH_EVAL_DATA_ROOT}
seed=${SEED}
num_gpus=8
question_batch_size=${UNIQUE_PROMPTS_PER_STEP}
per_device_batch_size=${PER_DEVICE_BATCH_SIZE}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}
steps_per_generation=${GRADIENT_ACCUMULATION_STEPS}
rollouts_per_question=${NUM_GENERATIONS}
training_trajectories_per_optimizer_step=${TRAJECTORIES_PER_STEP}
num_policy_iterations=${NUM_ITERATIONS}
grpo_epsilon=${GRPO_EPSILON}
entropy_coefficient=${ENTROPY_COEFFICIENT}
entropy_coefficient_note=historical_OPSD_TRL_GRPO_has_no_entropy_bonus
loss_aggregation=${LOSS_TYPE}
reward_scaling=${SCALE_REWARDS}
importance_sampling_level=${IMPORTANCE_SAMPLING_LEVEL}
vllm_is_correction=true
vllm_is_mode=${VLLM_IS_MODE}
vllm_is_cap=${VLLM_IS_CAP}
reference_kl_beta=0
learning_rate=${LR}
optimizer=adamw_torch
lr_scheduler=linear
warmup_steps=0
weight_decay=0
gradient_clip=0.1
max_prompt_length=${MAX_PROMPT_LENGTH}
max_response_length=${MAX_COMPLETION_LENGTH}
training_thinking=false
training_temperature=${ROLLOUT_TEMPERATURE}
training_top_p=${ROLLOUT_TOP_P}
training_top_k=${ROLLOUT_TOP_K}
total_training_steps=${SCHEDULER_HORIZON_STEPS}
physical_stop_step=${MAX_STEPS}
lora_rank=64
lora_alpha=128
lora_dropout=${LORA_DROPOUT}
evaluation_frequency=${EVAL_FREQUENCY}
evaluation_samples_per_question=${VAL_N}
evaluation_datasets=aime24,aime25,hmmt25,amc23,minerva
evaluation_thinking=true
evaluation_temperature=${EVAL_TEMPERATURE}
evaluation_top_p=${EVAL_TOP_P}
evaluation_top_k=${EVAL_TOP_K}
evaluation_max_new_tokens=${EVAL_MAX_NEW_TOKENS}
evaluation_tensor_parallel_size=8
evaluation_submission_mode=${EVAL_SUBMISSION_MODE}
evaluation_prompt_batch_size=${EVAL_PROMPT_BATCH_SIZE}
EOF
lock_protocol_file "${STATE_ROOT}/protocol.env" "${PROTOCOL_CANDIDATE}"
PROTOCOL_SHA256="$(sha256sum "${STATE_ROOT}/protocol.env" | awk '{print $1}')"

cat >"${STATE_ROOT}/parameters.env" <<EOF
method=grpo
framework=OPSD-TRL-Accelerate-DeepSpeed
code_commit=${CODE_COMMIT}
protocol_sha256=${PROTOCOL_SHA256}
run_name=${RUN_NAME}
run_root=${RUN_ROOT}
model_size=${MODEL_SIZE}
hardware=${HARDWARE}
model=${BASE_MODEL_DIR}
model_config_sha256=${MODEL_CONFIG_SHA256}
tokenizer_config_sha256=${TOKENIZER_CONFIG_SHA256}
training_dataset=${MATH_TRAIN_DATA}
training_dataset_sha256=${TRAIN_DATA_SHA256}
trainer=${OPSD_CODE_ROOT}/grpo_train.py
trainer_sha256=${TRAINER_SHA256}
training_metrics_jsonl=${METRICS_JSONL}
seed=${SEED}
num_gpus=8
question_batch_size=${UNIQUE_PROMPTS_PER_STEP}
per_device_batch_size=${PER_DEVICE_BATCH_SIZE}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}
steps_per_generation=${GRADIENT_ACCUMULATION_STEPS}
rollouts_per_question=${NUM_GENERATIONS}
training_trajectories_per_optimizer_step=${TRAJECTORIES_PER_STEP}
num_policy_iterations=${NUM_ITERATIONS}
grpo_epsilon=${GRPO_EPSILON}
entropy_coefficient=${ENTROPY_COEFFICIENT}
entropy_coefficient_note=historical_OPSD_TRL_GRPO_has_no_entropy_bonus
loss_aggregation=${LOSS_TYPE}
reward_scaling=${SCALE_REWARDS}
importance_sampling_level=${IMPORTANCE_SAMPLING_LEVEL}
vllm_is_correction=true
vllm_is_mode=${VLLM_IS_MODE}
vllm_is_cap=${VLLM_IS_CAP}
reference_kl_beta=0
learning_rate=${LR}
optimizer=adamw_torch
lr_scheduler=linear
warmup_steps=0
weight_decay=0
gradient_clip=0.1
max_prompt_length=${MAX_PROMPT_LENGTH}
max_response_length=${MAX_COMPLETION_LENGTH}
training_thinking=false
training_temperature=${ROLLOUT_TEMPERATURE}
training_top_p=${ROLLOUT_TOP_P}
training_top_k=${ROLLOUT_TOP_K}
total_training_steps=${SCHEDULER_HORIZON_STEPS}
physical_stop_step=${MAX_STEPS}
lora_rank=64
lora_alpha=128
lora_dropout=${LORA_DROPOUT}
evaluation_frequency=${EVAL_FREQUENCY}
evaluation_samples_per_question=${VAL_N}
evaluation_datasets=aime24,aime25,hmmt25,amc23,minerva
evaluation_thinking=true
evaluation_temperature=${EVAL_TEMPERATURE}
evaluation_top_p=${EVAL_TOP_P}
evaluation_top_k=${EVAL_TOP_K}
evaluation_max_new_tokens=${EVAL_MAX_NEW_TOKENS}
evaluation_tensor_parallel_size=8
evaluation_submission_mode=${EVAL_SUBMISSION_MODE}
evaluation_prompt_batch_size=${EVAL_PROMPT_BATCH_SIZE}
EOF

checkpoint_dir() {
  printf '%s/checkpoint-%s' "${TRAIN_OUTPUT_DIR}" "$1"
}

checkpoint_ready() {
  local dir
  dir="$(checkpoint_dir "$1")"
  [[ -s "${dir}/trainer_state.json" ]] &&
    [[ -s "${dir}/adapter_model.safetensors" || -s "${dir}/adapter_model.bin" ]] &&
    compgen -G "${dir}/global_step*" >/dev/null
}

result_complete() {
  local step="$1"
  local dataset
  for dataset in aime24 aime25 hmmt25 amc23 minerva; do
    PYTHONPATH="$(eval_pythonpath)" "${ENV_DIR}/bin/python" \
      "${REPO}/scripts/math/validate_math_eval.py" \
      "${RESULT_ROOT}/checkpoint-${step}/${dataset}.json" \
      --dataset "${dataset}" --samples "${VAL_N}" >/dev/null 2>&1 || return 1
  done
}

remove_checkpoint() {
  local step="$1"
  local dir
  dir="$(checkpoint_dir "${step}")"
  case "${dir}" in
    "${TRAIN_OUTPUT_DIR}"/checkpoint-[0-9]*) ;;
    *) echo "ERROR: refusing unsafe checkpoint deletion: ${dir}" >&2; return 2 ;;
  esac
  if [[ -d "${dir}" ]]; then
    rm -rf -- "${dir}"
    echo "Deleted evaluated checkpoint-${step}: ${dir}"
  fi
}

remove_older_checkpoints() {
  local current_step="$1"
  local dir step
  for dir in "${TRAIN_OUTPUT_DIR}"/checkpoint-*; do
    [[ -d "${dir}" ]] || continue
    step="${dir##*-}"
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if (( step < current_step )); then
      remove_checkpoint "${step}"
    fi
  done
}

wait_for_gpu_release() {
  local timeout_seconds="${1:-240}"
  local threshold_mib="${GPU_RELEASE_MEMORY_THRESHOLD_MIB:-4096}"
  local deadline=$((SECONDS + timeout_seconds))
  local max_used_mib
  while (( SECONDS < deadline )); do
    max_used_mib="$(
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null |
        awk 'BEGIN { max=0; ok=0 } /^[[:space:]]*[0-9]+/ { if ($1>max) max=$1; ok=1 } END { if (ok) print max }'
    )"
    if [[ "${max_used_mib}" =~ ^[0-9]+$ ]] && (( max_used_mib <= threshold_mib )); then
      echo "GPU workload released: max_used=${max_used_mib} MiB"
      return 0
    fi
    sleep 5
  done
  echo "ERROR: GPU workload did not release within ${timeout_seconds}s" >&2
  nvidia-smi >&2 || true
  return 1
}

start_adaptive_gpu_keepalive() {
  [[ "${ENABLE_ADAPTIVE_GPU_KEEPALIVE:-0}" == "1" ]] || return 0
  local keepalive_root="${STATE_ROOT}/gpu_keepalive"
  local keepalive_log="${keepalive_root}/workers.log"
  local gpu pid
  mkdir -p "${keepalive_root}"
  KEEPALIVE_STOP_FILE="${keepalive_root}/stop"
  rm -f -- "${KEEPALIVE_STOP_FILE}"
  : >"${keepalive_log}"

  for gpu in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${ENV_DIR}/bin/python" "${KEEPALIVE_SCRIPT}" \
        --stop-file "${KEEPALIVE_STOP_FILE}" \
        --parent-pid "$$" \
        --physical-gpu "${gpu}" \
        --minimum-utilization "${KEEPALIVE_MIN_UTILIZATION:-38}" \
        --burst-seconds "${KEEPALIVE_BURST_SECONDS:-0.7}" \
        --idle-seconds "${KEEPALIVE_IDLE_SECONDS:-0.8}" \
        --startup-delay-seconds "${KEEPALIVE_STARTUP_DELAY_SECONDS:-30}" \
        --matrix-size "${KEEPALIVE_MATRIX_SIZE:-2048}" \
        --minimum-used-memory-mib "${KEEPALIVE_MIN_USED_MEMORY_MIB:-4096}" \
        >>"${keepalive_log}" 2>&1 &
    pid=$!
    KEEPALIVE_PIDS+=("${pid}")
  done
  printf '%s\n' "${KEEPALIVE_PIDS[@]}" >"${keepalive_root}/worker_pids"
  echo "Adaptive GPU keepalive enabled; this does not change model updates or samples"
}

validate_grpo_inputs() {
  validate_inputs
  PYTHONPATH="$(distil_pythonpath)" "${ENV_DIR}/bin/python" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["MATH_TRAIN_DATA"])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 758:
    raise RuntimeError(f"expected 758 Math rows, found {len(rows)}")
for index, row in enumerate(rows):
    for key in ("problem", "answer"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise RuntimeError(f"row {index} has invalid {key}")
print(f"GRPO local dataset contract: PASS ({len(rows)} rows)")
PY
}

launch_phase() {
  local stop_after_step="$1"
  local port="$2"
  local command=(
    "${ENV_DIR}/bin/accelerate" launch
    --config_file "${OPSD_CODE_ROOT}/accelerate.yaml"
    --num_processes 8
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --main_process_port "${port}"
    "${OPSD_CODE_ROOT}/grpo_train.py"
    --model_name_or_path "${BASE_MODEL_DIR}"
    --dataset_name "${MATH_TRAIN_DATA}"
    --learning_rate "${LR}"
    --optim adamw_torch
    --lr_scheduler_type linear
    --warmup_steps 0
    --weight_decay 0
    --max_grad_norm 0.1
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --gradient_checkpointing
    --output_dir "${CHECKPOINT_ROOT}"
    --run_config "${RUN_NAME}"
    --max_steps "${SCHEDULER_HORIZON_STEPS}"
    --save_strategy no
    --selected_checkpoint_steps "${SAVE_STEPS}"
    --stop_after_step "${stop_after_step}"
    --auto_resume true
    --save_final_model false
    --logging_steps 1
    --eval_strategy no
    --dataloader_num_workers 0
    --max_prompt_length "${MAX_PROMPT_LENGTH}"
    --max_completion_length "${MAX_COMPLETION_LENGTH}"
    --attn_implementation flash_attention_2
    --torch_dtype bfloat16
    --enable_thinking false
    --num_generations "${NUM_GENERATIONS}"
    --steps_per_generation "${GRADIENT_ACCUMULATION_STEPS}"
    --num_iterations "${NUM_ITERATIONS}"
    --epsilon "${GRPO_EPSILON}"
    --epsilon_high "${GRPO_EPSILON}"
    --beta 0
    --loss_type "${LOSS_TYPE}"
    --scale_rewards "${SCALE_REWARDS}"
    --importance_sampling_level "${IMPORTANCE_SAMPLING_LEVEL}"
    --temperature "${ROLLOUT_TEMPERATURE}"
    --top_p "${ROLLOUT_TOP_P}"
    --top_k "${ROLLOUT_TOP_K}"
    --use_vllm
    --vllm_mode colocate
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --vllm_max_model_length 18432
    --vllm_tensor_parallel_size 1
    --vllm_importance_sampling_correction true
    --vllm_importance_sampling_mode "${VLLM_IS_MODE}"
    --vllm_importance_sampling_cap "${VLLM_IS_CAP}"
    --use_peft
    --lora_r 64
    --lora_alpha 128
    --lora_dropout "${LORA_DROPOUT}"
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --seed "${SEED}"
    --data_seed "${SEED}"
    --report_to none
  )

  echo "Starting OPSD/TRL GRPO training phase through step ${stop_after_step}"
  set +e
  (
    cd "${OPSD_CODE_ROOT}"
    PYTHONPATH="$(distil_pythonpath)" "${command[@]}"
  ) 2>&1 | tee -a "${LOG_ROOT}/training.log"
  local status=${PIPESTATUS[0]}
  set -e
  (( status == 0 )) || return "${status}"
  checkpoint_ready "${stop_after_step}" || {
    echo "ERROR: phase exited without a resumable checkpoint-${stop_after_step}" >&2
    return 4
  }
}

evaluate_step() (
  set -Eeuo pipefail
  local step="$1"
  local checkpoint result_dir
  checkpoint="$(checkpoint_dir "${step}")"
  result_dir="${RESULT_ROOT}/checkpoint-${step}"
  mkdir -p "${result_dir}"
  echo "Evaluating OPSD/TRL GRPO checkpoint-${step} on five datasets with N=${VAL_N}"
  PYTHONPATH="$(eval_pythonpath)" \
  PYTHON_BIN="${ENV_DIR}/bin/python" \
  MODEL_DIR="${BASE_MODEL_DIR}" \
  TOKENIZER_DIR="${BASE_MODEL_DIR}" \
  LORA_ADAPTER_DIR="${checkpoint}" \
  MODEL_SIZE="${MODEL_SIZE}" \
  OUTPUT_DIR="${result_dir}" \
  VAL_N="${VAL_N}" \
  TENSOR_PARALLEL_SIZE=8 \
  EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.90}" \
  EVAL_SUBMISSION_MODE="${EVAL_SUBMISSION_MODE}" \
  EVAL_PROMPT_BATCH_SIZE="${EVAL_PROMPT_BATCH_SIZE}" \
  EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS}" \
  MATH_EVAL_DATA_ROOT="${MATH_EVAL_DATA_ROOT}" \
    bash "${REPO}/scripts/math/eval_sr_opsd_verl_math.sh"
  result_complete "${step}"
)

echo "============================================================"
echo "Qwen3-${MODEL_SIZE^^} GRPO OPSD/TRL train/evaluate pipeline"
echo "host=$(hostname)"
echo "hardware=${HARDWARE}"
echo "framework=OPSD/TRL 0.26 + Accelerate + DeepSpeed + LoRA"
echo "training=${MAX_STEPS} physical steps; scheduler horizon=${SCHEDULER_HORIZON_STEPS}; eval every ${EVAL_FREQUENCY}"
echo "sampling=${UNIQUE_PROMPTS_PER_STEP} unique prompts x ${NUM_GENERATIONS} rollouts = ${TRAJECTORIES_PER_STEP} trajectories per optimizer step"
echo "objective=GRPO epsilon=${GRPO_EPSILON}; one iteration; ${LOSS_TYPE} token mean; group-normalized rewards"
echo "entropy_coefficient=${ENTROPY_COEFFICIENT} (historical OPSD/TRL GRPO has no entropy-bonus term)"
echo "evaluation=five Math datasets; thinking; N=${VAL_N}; max_new_tokens=${EVAL_MAX_NEW_TOKENS}; submission=${EVAL_SUBMISSION_MODE}"
echo "checkpoint_policy=retain current resume point only; delete after next checkpoint; delete final after eval"
echo "output=${RUN_ROOT}"
echo "online_loggers=disabled"
echo "============================================================"

runtime_preflight
validate_grpo_inputs
if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "PRECHECK: PASS"
  exit 0
fi
start_adaptive_gpu_keepalive

if [[ -f "${STATE_ROOT}/complete" ]]; then
  echo "SKIP: pipeline is already complete: ${RUN_ROOT}"
  exit 0
fi

phase_index=0
for step in $(seq "${EVAL_FREQUENCY}" "${EVAL_FREQUENCY}" "${MAX_STEPS}"); do
  if result_complete "${step}"; then
    echo "SKIP complete evaluation at checkpoint-${step}"
  else
    if checkpoint_ready "${step}"; then
      echo "Reusing resumable checkpoint-${step}"
    else
      launch_phase "${step}" "$((MAIN_PROCESS_PORT + phase_index))"
      wait_for_gpu_release 300
    fi
    remove_older_checkpoints "${step}"
    evaluate_step "${step}"
    wait_for_gpu_release 300
  fi
  if checkpoint_ready "${step}"; then
    remove_older_checkpoints "${step}"
  fi
  phase_index=$((phase_index + 1))
done

result_complete "${MAX_STEPS}"
remove_checkpoint "${MAX_STEPS}"
touch "${STATE_ROOT}/complete"
echo "COMPLETE: Qwen3-${MODEL_SIZE^^} GRPO trained through step ${MAX_STEPS} and all N=${VAL_N} evaluations are complete"
echo "results=${RESULT_ROOT}"
