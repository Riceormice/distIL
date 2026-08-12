#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_distil_a800.sh"

MAX_STEPS=100
SCHEDULER_HORIZON_STEPS=420
VAL_N=16
SAVE_STEPS="$(seq -s, 5 5 100)"
SEED=0
LR=5e-6
GRADIENT_ACCUMULATION_STEPS=1
ROLLOUT_N=8
ROLLOUT_TEMPERATURE=0.7
TOP_P=0.95
TOP_K=20
MAX_COMPLETION_LENGTH=16384
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29620}"
RUN_NAME="opsd-4b-seed0-lr5e-6-bs1-ga1-steps100-sched420-beta0-clip0.05-topk100-temp0.7-tok16384-eval5-n16-a800"

RUN_ROOT="${OUTPUT_ROOT}/opsd/${RUN_NAME}"
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"
TRAIN_OUTPUT_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"
RESULT_ROOT="${RUN_ROOT}/evaluations"
LOG_ROOT="${RUN_ROOT}/logs"
STATE_ROOT="${RUN_ROOT}/state"
METRICS_JSONL="${RUN_ROOT}/training_metrics.jsonl"
mkdir -p "${TRAIN_OUTPUT_DIR}" "${RESULT_ROOT}" "${LOG_ROOT}" "${STATE_ROOT}"

exec 9>"${STATE_ROOT}/pipeline.lock"
flock -n 9 || { echo "ERROR: OPSD pipeline is already running: ${RUN_ROOT}" >&2; exit 3; }
LAUNCH_LOG="${LOG_ROOT}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1
export SDPO_METRICS_JSONL="${METRICS_JSONL}"

cat >"${STATE_ROOT}/parameters.env" <<EOF
method=opsd
framework=distIL-TRL
model=${BASE_MODEL_DIR}
seed=${SEED}
per_device_batch_size=1
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}
rollouts_per_question=${ROLLOUT_N}
learning_rate=${LR}
lr_scheduler=linear
warmup_steps=0
weight_decay=0
gradient_clip=0.1
max_response_length=${MAX_COMPLETION_LENGTH}
training_temperature=${ROLLOUT_TEMPERATURE}
training_top_p=${TOP_P}
training_top_k=${TOP_K}
total_training_steps=${SCHEDULER_HORIZON_STEPS}
physical_stop_step=${MAX_STEPS}
evaluation_frequency=5
evaluation_samples_per_question=${VAL_N}
evaluation_datasets=aime24,aime25,hmmt25,amc23,minerva
EOF

checkpoint_dir() {
  printf '%s/checkpoint-%s' "${TRAIN_OUTPUT_DIR}" "$1"
}

checkpoint_ready() {
  local dir
  dir="$(checkpoint_dir "$1")"
  [[ -s "${dir}/adapter_model.safetensors" || -s "${dir}/adapter_model.bin" ]] &&
    compgen -G "${dir}/global_step*" >/dev/null
}

result_complete() {
  local step="$1" dataset
  for dataset in aime24 aime25 hmmt25 amc23 minerva; do
    PYTHONPATH="$(eval_pythonpath)" "${ENV_DIR}/bin/python" \
      "${REPO}/scripts/math/validate_math_eval.py" \
      "${RESULT_ROOT}/checkpoint-${step}/${dataset}.json" \
      --dataset "${dataset}" --samples "${VAL_N}" >/dev/null 2>&1 || return 1
  done
}

remove_checkpoint() {
  local step="$1" dir
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

launch_phase() {
  local stop_after_step="$1" port="$2"
  local command=(
    "${ENV_DIR}/bin/accelerate" launch
    --config_file "${BASELINE_OPSD}/accelerate.yaml"
    --num_processes 8
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --main_process_port "${port}"
    "${BASELINE_OPSD}/opsd_train.py"
    --model_name_or_path "${BASE_MODEL_DIR}"
    --dataset_name "${MATH_TRAIN_DATA}"
    --loss_mode jsd
    --learning_rate "${LR}"
    --lr_scheduler_type linear
    --warmup_steps 0
    --weight_decay 0
    --max_grad_norm 0.1
    --per_device_train_batch_size 1
    --gradient_checkpointing
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --output_dir "${CHECKPOINT_ROOT}"
    --run_config "${RUN_NAME}"
    --max_steps "${SCHEDULER_HORIZON_STEPS}"
    --save_strategy no
    --save_steps 100
    --selected_checkpoint_steps "${SAVE_STEPS}"
    --stop_after_step "${stop_after_step}"
    --logging_steps 1
    --eval_strategy no
    --max_completion_length "${MAX_COMPLETION_LENGTH}"
    --attn_implementation flash_attention_2
    --torch_dtype bfloat16
    --max_length 20000
    --beta 0
    --use_vllm
    --vllm_mode colocate
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
    --vllm_tensor_parallel_size 1
    --use_peft
    --lora_r 64
    --lora_alpha 128
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --num_generations "${ROLLOUT_N}"
    --temperature "${ROLLOUT_TEMPERATURE}"
    --top_p "${TOP_P}"
    --top_k "${TOP_K}"
    --top_k_loss 100
    --lmbda 1
    --fixed_teacher
    --jsd_token_clip 0.05
    --seed "${SEED}"
    --data_seed "${SEED}"
    --wandb_project local-only
    --report_to none
  )

  echo "Starting OPSD training phase through step ${stop_after_step}"
  set +e
  (
    cd "${BASELINE_OPSD}"
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
  local step="$1" checkpoint result_dir
  checkpoint="$(checkpoint_dir "${step}")"
  result_dir="${RESULT_ROOT}/checkpoint-${step}"
  mkdir -p "${result_dir}"
  PYTHONPATH="$(eval_pythonpath)" \
  PYTHON_BIN="${ENV_DIR}/bin/python" \
  MODEL_DIR="${BASE_MODEL_DIR}" \
  LORA_ADAPTER_DIR="${checkpoint}" \
  MODEL_SIZE=4b \
  OUTPUT_DIR="${result_dir}" \
  VAL_N="${VAL_N}" \
  TENSOR_PARALLEL_SIZE=8 \
  EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.90}" \
  MATH_EVAL_DATA_ROOT="${MATH_EVAL_DATA_ROOT}" \
    bash "${REPO}/scripts/math/eval_sr_opsd_verl_math.sh"
  result_complete "${step}"
)

remove_older_checkpoints() {
  local current_step="$1" dir step
  for dir in "${TRAIN_OUTPUT_DIR}"/checkpoint-*; do
    [[ -d "${dir}" ]] || continue
    step="${dir##*-}"
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if (( step < current_step )); then
      remove_checkpoint "${step}"
    fi
  done
}

echo "============================================================"
echo "Qwen3-4B OPSD A800 train/evaluate pipeline"
echo "host=$(hostname)"
echo "training=100 physical steps; scheduler horizon=420; evaluation every 5 steps"
echo "evaluation=five math datasets; thinking; N=16; TP=8"
echo "output=${RUN_ROOT}"
echo "online_loggers=disabled"
echo "============================================================"

runtime_preflight
validate_inputs
if [[ -f "${STATE_ROOT}/complete" ]]; then
  echo "SKIP: pipeline is already complete: ${RUN_ROOT}"
  exit 0
fi

phase_index=0
for step in $(seq 5 5 "${MAX_STEPS}"); do
  if result_complete "${step}"; then
    echo "SKIP complete evaluation at checkpoint-${step}"
  else
    if checkpoint_ready "${step}"; then
      echo "Reusing resumable checkpoint-${step}"
    else
      launch_phase "${step}" "$((MAIN_PROCESS_PORT + phase_index))"
      wait_for_gpu_release 180
    fi
    remove_older_checkpoints "${step}"
    evaluate_step "${step}"
    wait_for_gpu_release 180
  fi
  if checkpoint_ready "${step}"; then
    remove_older_checkpoints "${step}"
  fi
  phase_index=$((phase_index + 1))
done

result_complete 100
remove_checkpoint 100
touch "${STATE_ROOT}/complete"
echo "COMPLETE: Qwen3-4B OPSD trained through step 100 and all 20 checkpoints were evaluated at N=16"
echo "results=${RESULT_ROOT}"
