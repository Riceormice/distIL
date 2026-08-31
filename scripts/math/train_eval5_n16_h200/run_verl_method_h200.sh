#!/usr/bin/env bash
set -Eeuo pipefail

METHOD="${1:?Usage: $0 grpo|sdpo|sr_opsd}"
case "${METHOD}" in
  grpo|sdpo|sr_opsd) ;;
  *) echo "ERROR: METHOD must be grpo, sdpo, or sr_opsd" >&2; exit 2 ;;
esac

ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-${ROOT}/code/distIL-sr-opsd-renyi}"
UNIFIED_ENV_ACTIVATE="${REPO}/scripts/math/unified_env/activate_unified_math_env.sh"
source "${UNIFIED_ENV_ACTIVATE}" verl
source "${REPO}/scripts/math/lock_protocol.sh"
PYTHON_BIN="${ENV_DIR}/bin/python"
export SDPO_SHARED_CHECKPOINT_STORE="${SDPO_SHARED_CHECKPOINT_STORE-/media/vlm-ckp-fileset/ylong/sdpo/shared_checkpoint_bases/v1}"
CHECKPOINT_STORAGE_MODE=plain
[[ -z "${SDPO_SHARED_CHECKPOINT_STORE}" ]] || CHECKPOINT_STORAGE_MODE=lossless_shared_v1
MODEL_SIZE="${MODEL_SIZE:-8b}"
HARDWARE="${HARDWARE:-h200}"
case "${MODEL_SIZE}" in
  4b) DEFAULT_MODEL_PATH=/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-4B ;;
  8b) DEFAULT_MODEL_PATH=/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-8B ;;
  *) echo "ERROR: MODEL_SIZE must be 4b or 8b" >&2; exit 2 ;;
esac
case "${HARDWARE}" in
  a800|h200) ;;
  *) echo "ERROR: HARDWARE must be a800 or h200" >&2; exit 2 ;;
esac
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
MATH_EVAL_DATA_ROOT="${MATH_EVAL_DATA_ROOT:-${ROOT}/data/math_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/math_train_eval5_n16_h200_20260812}"
SITE_PACKAGES="${ENV_DIR}/lib/python3.11/site-packages"

# For SR-OPSD, SELF_REFERENCE_WEIGHT is the self-reference coefficient
# (the paper's alpha). DIVERGENCE_ALPHA remains the fixed Forward-Renyi
# implementation parameter and is intentionally not swept here.
DIVERGENCE_ALPHA="${DIVERGENCE_ALPHA:-0.25}"
RENYI_ORDER="${RENYI_ORDER:-0.95}"
SELF_REFERENCE_WEIGHT="${SELF_REFERENCE_WEIGHT:-0.9}"

VAL_N=16
MAX_STEPS=100
SCHEDULER_HORIZON_STEPS=420
TRAIN_BATCH_SIZE=8
PPO_MINI_BATCH_SIZE=8
ROLLOUT_N=8

# GRPO currently fails to terminate reliably under the historical 38,912-token
# 8B evaluation budget. Keep the legacy budget for the other 8B methods while
# capping only GRPO at the 16,384-token training response limit.
case "${MODEL_SIZE}:${METHOD}" in
  8b:grpo) EVAL_MAX_NEW_TOKENS=16384 ;;
  8b:*) EVAL_MAX_NEW_TOKENS=38912 ;;
  4b:*) EVAL_MAX_NEW_TOKENS=16384 ;;
esac
case "${MODEL_SIZE}" in
  4b)
    EVAL_TEMPERATURE=0.7
    EVAL_TOP_P=0.95
    EVAL_TOP_K=20
    ;;
  8b)
    EVAL_TEMPERATURE=1.0
    EVAL_TOP_P=1.0
    EVAL_TOP_K=-1
    ;;
esac
EVAL_SUBMISSION_MODE="${EVAL_SUBMISSION_MODE:-legacy_all_prompts}"
case "${EVAL_SUBMISSION_MODE}" in
  legacy_all_prompts) EVAL_PROMPT_BATCH_SIZE=0 ;;
  chunked) EVAL_PROMPT_BATCH_SIZE="${EVAL_PROMPT_BATCH_SIZE:-8}" ;;
  *) echo "ERROR: unsupported EVAL_SUBMISSION_MODE=${EVAL_SUBMISSION_MODE}" >&2; exit 2 ;;
esac

case "${METHOD}" in
  grpo)
    METHOD_LABEL=GRPO
    METHOD_DIR=grpo
    RUN_NAME="grpo-${MODEL_SIZE}-seed0-native-verl-lr5e-6-trainbs8-mbs8-rolloutn8-eps0.2-temp0.7-tok16384-steps100-sched420-eval5-n16-${HARDWARE}"
    METHOD_DESCRIPTION="GRPO; epsilon=0.2; group-normalized advantages"
    EFFECTIVE_DIVERGENCE_ALPHA=NA
    EFFECTIVE_RENYI_ORDER=NA
    EFFECTIVE_SELF_REFERENCE_WEIGHT=NA
    ;;
  sdpo)
    METHOD_LABEL="SDPO (RKL)"
    METHOD_DIR=sdpo
    RUN_NAME="sdpo-${MODEL_SIZE}-seed0-native-verl-rkl-ema0.05-lr5e-6-trainbs8-mbs8-rolloutn8-topk100-tailFalse-clip0.05-temp0.7-tok16384-steps100-sched420-eval5-n16-${HARDWARE}"
    METHOD_DESCRIPTION="Reverse KL; EMA teacher=0.05; no frozen-reference anchoring"
    EFFECTIVE_DIVERGENCE_ALPHA=1.0
    EFFECTIVE_RENYI_ORDER=NA
    EFFECTIVE_SELF_REFERENCE_WEIGHT=0.0
    ;;
  sr_opsd)
    METHOD_LABEL=SR-OPSD
    METHOD_DIR=sr_opsd
    RUN_NAME="sr-opsd-${MODEL_SIZE}-seed0-native-verl-forward-renyi-rho${RENYI_ORDER}-refw${SELF_REFERENCE_WEIGHT}-sync0-ema0.05-lr5e-6-trainbs8-mbs8-rolloutn8-topk100-tailFalse-clip0.05-temp0.7-tok16384-steps100-sched420-eval5-n16-${HARDWARE}"
    METHOD_DESCRIPTION="Forward Renyi rho=${RENYI_ORDER}; self-reference=${SELF_REFERENCE_WEIGHT}; frozen reference sync=0"
    EFFECTIVE_DIVERGENCE_ALPHA=${DIVERGENCE_ALPHA}
    EFFECTIVE_RENYI_ORDER=${RENYI_ORDER}
    EFFECTIVE_SELF_REFERENCE_WEIGHT=${SELF_REFERENCE_WEIGHT}
    ;;
esac

RUN_ROOT="${OUTPUT_ROOT}/${METHOD_DIR}/${RUN_NAME}"
TRAIN_OUTPUT_ROOT="${RUN_ROOT}/native"
RUN_DIR="${TRAIN_OUTPUT_ROOT}/checkpoints/${RUN_NAME}"
RESULT_ROOT="${RUN_ROOT}/evaluations"
MERGED_ROOT="${RUN_ROOT}/merged"
LOG_ROOT="${RUN_ROOT}/logs"
STATE_ROOT="${RUN_ROOT}/state"
KEEPALIVE_SCRIPT="${REPO}/scripts/math/adaptive_gpu_keepalive.py"
KEEPALIVE_PIDS=()
KEEPALIVE_STOP_FILE=""

export PYTHONPATH="${REPO}/SDPO:${REPO}"
export PYTHON_BIN
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
unset SETUPTOOLS_USE_DISTUTILS
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/media/vlm-ckp-fileset/ylong/sdpo/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/media/vlm-ckp-fileset/ylong/sdpo/cache/datasets}"
export VLLM_USE_V1=1
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export SWANLAB_MODE=offline
export SDPO_SWANLAB_MODE=offline
export SWANLAB_DISABLED=1
export WANDB_MODE=offline
unset SWANLAB_API_KEY SWANLAB_WORKSPACE SWANLAB_PROJECT
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

mkdir -p "${LOG_ROOT}" "${STATE_ROOT}"
LAUNCH_LOG="${LOG_ROOT}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1
report_error() {
  local status=$?
  local line="${BASH_LINENO[0]:-unknown}"
  local command="${BASH_COMMAND:-unknown}"
  trap - ERR
  echo "ERROR: native VERL pipeline failed: status=${status} line=${line} command=${command}" >&2
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

test -x "${PYTHON_BIN}"
test -f "${ENV_DIR}/.math-env-complete"
test -x "${SITE_PACKAGES}/torch/bin/torch_shm_manager"
test -f "${SITE_PACKAGES}/flash_attn/__init__.py"
compgen -G "${SITE_PACKAGES}/flash_attn_2_cuda*.so" >/dev/null
test -f "${SITE_PACKAGES}/transformers/__init__.py"
test -f "${SITE_PACKAGES}/tokenizers/__init__.py"
test -f "${SITE_PACKAGES}/vllm/version.py"
compgen -G "${SITE_PACKAGES}/vllm/_C*.so" >/dev/null
test -f "${MODEL_PATH}/config.json"
test -f "${MODEL_PATH}/tokenizer_config.json"
test -f "${REPO}/SDPO/datasets/math_probs/train.json"
test -f "${REPO}/SDPO/datasets/math_probs/test.json"
test -f "${REPO}/SDPO/run_local_math_verl.sh"
test -f "${REPO}/scripts/math/eval_sr_opsd_verl_math.sh"
test -f "${REPO}/scripts/math/validate_math_eval.py"
test -f "${KEEPALIVE_SCRIPT}"
mkdir -p "${RUN_DIR}" "${RESULT_ROOT}" "${MERGED_ROOT}"

TRAIN_DATA_SHA256="$(sha256sum "${REPO}/SDPO/datasets/math_probs/train.json" | awk '{print $1}')"
MODEL_CONFIG_SHA256="$(sha256sum "${MODEL_PATH}/config.json" | awk '{print $1}')"
TOKENIZER_CONFIG_SHA256="$(sha256sum "${MODEL_PATH}/tokenizer_config.json" | awk '{print $1}')"
CODE_COMMIT="$(git -C "${REPO}" rev-parse HEAD 2>/dev/null || printf 'unknown')"

exec 9>"${STATE_ROOT}/pipeline.lock"
flock -n 9 || { echo "ERROR: ${METHOD_LABEL} pipeline is already running: ${RUN_ROOT}" >&2; exit 3; }

PROTOCOL_CANDIDATE="${STATE_ROOT}/protocol.env.candidate.$$"
cat >"${PROTOCOL_CANDIDATE}" <<EOF
method=${METHOD}
framework=SDPO-native-VERL
model_size=${MODEL_SIZE}
hardware=${HARDWARE}
model_path=${MODEL_PATH}
model_config_sha256=${MODEL_CONFIG_SHA256}
tokenizer_config_sha256=${TOKENIZER_CONFIG_SHA256}
training_dataset_sha256=${TRAIN_DATA_SHA256}
evaluation_data_root=${MATH_EVAL_DATA_ROOT}
seed=0
num_gpus=8
train_batch_size=${TRAIN_BATCH_SIZE}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
ppo_micro_batch_size_per_gpu=1
ppo_epochs=1
rollouts_per_question=${ROLLOUT_N}
learning_rate=5e-6
lr_scheduler=linear
warmup_steps=0
weight_decay=0
gradient_clip=0.1
entropy_coefficient=1e-5
max_prompt_length=2048
max_response_length=16384
max_reprompt_length=16384
max_model_length=32768
training_temperature=0.7
training_top_p=0.95
training_top_k=20
total_training_steps=${SCHEDULER_HORIZON_STEPS}
physical_stop_step=${MAX_STEPS}
lora_rank=64
lora_alpha=128
evaluation_frequency=5
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
divergence_alpha=${EFFECTIVE_DIVERGENCE_ALPHA}
renyi_order=${EFFECTIVE_RENYI_ORDER}
self_reference_weight=${EFFECTIVE_SELF_REFERENCE_WEIGHT}
method_description=${METHOD_DESCRIPTION}
EOF
lock_protocol_file "${STATE_ROOT}/protocol.env" "${PROTOCOL_CANDIDATE}"
PROTOCOL_SHA256="$(sha256sum "${STATE_ROOT}/protocol.env" | awk '{print $1}')"

cat >"${STATE_ROOT}/parameters.env" <<EOF
method=${METHOD}
framework=SDPO-native-VERL
code_commit=${CODE_COMMIT}
checkpoint_storage=${CHECKPOINT_STORAGE_MODE}
shared_checkpoint_store=${SDPO_SHARED_CHECKPOINT_STORE}
protocol_sha256=${PROTOCOL_SHA256}
model_size=${MODEL_SIZE}
hardware=${HARDWARE}
model=${MODEL_PATH}
model_config_sha256=${MODEL_CONFIG_SHA256}
tokenizer_config_sha256=${TOKENIZER_CONFIG_SHA256}
training_dataset=${REPO}/SDPO/datasets/math_probs/train.json
training_dataset_sha256=${TRAIN_DATA_SHA256}
seed=0
num_gpus=8
train_batch_size=${TRAIN_BATCH_SIZE}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
ppo_micro_batch_size_per_gpu=1
ppo_epochs=1
rollouts_per_question=${ROLLOUT_N}
learning_rate=5e-6
lr_scheduler=linear
warmup_steps=0
weight_decay=0
gradient_clip=0.1
entropy_coefficient=1e-5
max_prompt_length=2048
max_response_length=16384
max_model_length=32768
training_temperature=0.7
training_top_p=0.95
training_top_k=20
total_training_steps=${SCHEDULER_HORIZON_STEPS}
physical_stop_step=${MAX_STEPS}
evaluation_frequency=5
evaluation_samples_per_question=${VAL_N}
evaluation_datasets=aime24,aime25,hmmt25,amc23,minerva
evaluation_thinking=true
evaluation_temperature=${EVAL_TEMPERATURE}
evaluation_top_p=${EVAL_TOP_P}
evaluation_top_k=${EVAL_TOP_K}
evaluation_submission_mode=${EVAL_SUBMISSION_MODE}
evaluation_prompt_batch_size=${EVAL_PROMPT_BATCH_SIZE}
evaluation_max_new_tokens=${EVAL_MAX_NEW_TOKENS}
evaluation_tensor_parallel_size=8
evaluation_tokenizer=${MODEL_PATH}
divergence_alpha=${EFFECTIVE_DIVERGENCE_ALPHA}
renyi_order=${EFFECTIVE_RENYI_ORDER}
self_reference_weight=${EFFECTIVE_SELF_REFERENCE_WEIGHT}
method_description=${METHOD_DESCRIPTION}
EOF

checkpoint_dir() {
  printf '%s/global_step_%s' "${RUN_DIR}" "$1"
}

checkpoint_ready() {
  local dir
  dir="$(checkpoint_dir "$1")"
  [[ -d "${dir}/actor" && -s "${dir}/data.pt" ]]
}

result_complete() {
  local step="$1"
  local dataset
  for dataset in aime24 aime25 hmmt25 amc23 minerva; do
    "${PYTHON_BIN}" "${REPO}/scripts/math/validate_math_eval.py" \
      "${RESULT_ROOT}/checkpoint-${step}/${dataset}.json" \
      --dataset "${dataset}" --samples "${VAL_N}" >/dev/null 2>&1 || return 1
  done
}

remove_native_checkpoint() {
  local step="$1"
  local dir
  dir="$(checkpoint_dir "${step}")"
  case "${dir}" in
    "${RUN_DIR}"/global_step_[0-9]*) ;;
    *) echo "ERROR: refusing unsafe checkpoint deletion: ${dir}" >&2; return 2 ;;
  esac
  if [[ -d "${dir}" ]]; then
    rm -rf -- "${dir}"
    echo "Deleted evaluated native checkpoint step ${step}: ${dir}"
  fi
}

wait_for_gpu_release() {
  local deadline=$((SECONDS + ${1:-240}))
  local threshold_mib="${GPU_RELEASE_MEMORY_THRESHOLD_MIB:-2048}"
  local max_used_mib
  while (( SECONDS < deadline )); do
    max_used_mib="$(
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null |
        awk '
          BEGIN { max = 0; valid = 0 }
          /^[[:space:]]*[0-9]+[[:space:]]*$/ {
            value = $1 + 0
            if (value > max) max = value
            valid = 1
          }
          END { if (valid) print max }
        '
    )"
    if [[ "${max_used_mib}" =~ ^[0-9]+$ ]] && (( max_used_mib <= threshold_mib )); then
      echo "GPU workload released: max_used=${max_used_mib} MiB, threshold=${threshold_mib} MiB"
      return 0
    fi
    sleep 5
  done
  echo "ERROR: GPU workload did not release in time (threshold=${threshold_mib} MiB)" >&2
  nvidia-smi >&2 || true
  return 1
}

start_adaptive_gpu_keepalive() {
  [[ "${ENABLE_ADAPTIVE_GPU_KEEPALIVE:-0}" == "1" ]] || return 0

  local -a visible_gpus
  IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
  if (( ${#visible_gpus[@]} != 8 )); then
    echo "ERROR: keepalive expects 8 visible GPUs, got ${CUDA_VISIBLE_DEVICES}" >&2
    return 2
  fi

  local keepalive_root="${STATE_ROOT}/gpu_keepalive"
  local keepalive_log="${keepalive_root}/workers.log"
  local gpu pid
  mkdir -p "${keepalive_root}"
  KEEPALIVE_STOP_FILE="${keepalive_root}/stop"
  rm -f -- "${KEEPALIVE_STOP_FILE}"
  : >"${keepalive_log}"

  for gpu in "${visible_gpus[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PYTHON_BIN}" "${KEEPALIVE_SCRIPT}" \
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

  sleep 1
  for pid in "${KEEPALIVE_PIDS[@]}"; do
    kill -0 "${pid}" 2>/dev/null || {
      echo "ERROR: adaptive GPU keepalive worker ${pid} exited during startup" >&2
      tail -n 80 "${keepalive_log}" >&2 || true
      return 3
    }
  done
  printf '%s\n' "${KEEPALIVE_PIDS[@]}" >"${keepalive_root}/worker_pids"
  echo "Adaptive GPU keepalive enabled: target>=${KEEPALIVE_MIN_UTILIZATION:-38}%"
  echo "keepalive_log=${keepalive_log}"
}

runtime_preflight() {
  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  [[ "${gpu_count}" == "8" ]] || { echo "ERROR: expected 8 GPUs, found ${gpu_count}" >&2; exit 2; }
  case "${HARDWARE}" in
    h200)
      if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Ev 'H200|H20Z' >/dev/null; then
        echo "ERROR: this launcher requires eight H200/H20Z GPUs" >&2
        exit 2
      fi
      ;;
    a800)
      if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Ev 'A800' >/dev/null; then
        echo "ERROR: this launcher requires eight A800 GPUs" >&2
        exit 2
      fi
      ;;
  esac
  nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader

  env -u PYTHONHOME -u CONDA_PREFIX \
    PYTHONPATH="${REPO}/SDPO:${REPO}" \
    "${PYTHON_BIN}" "${REPO}/scripts/math/unified_env/verify_unified_math_env.py" \
      --profile verl --prefix "${ENV_DIR}" --repo "${REPO}" --gpu-smoke

  HARDWARE="${HARDWARE}" "${PYTHON_BIN}" - <<'PY'
import importlib
import os
import torch
import torch.multiprocessing as mp
from flash_attn import flash_attn_func

assert torch.cuda.device_count() == 8, torch.cuda.device_count()
expected_capability = (8, 0) if os.environ["HARDWARE"] == "a800" else (9, 0)
for index in range(8):
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    assert capability == expected_capability, (index, name, capability, expected_capability)
for module in ("flash_attn", "ray", "ray._common.utils", "transformers", "vllm", "verl"):
    imported = importlib.import_module(module)
    print(f"{module}: {getattr(imported, '__file__', '<namespace>')}")
import tokenizers
print(f"tokenizers={tokenizers.__version__}: {tokenizers.__file__}")
mp.set_sharing_strategy("file_system")
torch.zeros(1).share_memory_()
q = torch.randn((1, 16, 4, 64), device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, q, q, causal=True)
torch.cuda.synchronize()
assert out.shape == q.shape
print(f"torch={torch.__version__}")
print(f"{os.environ['HARDWARE'].upper()} BF16 FlashAttention smoke test: PASS")
PY

  MATH_EVAL_DATA_ROOT="${MATH_EVAL_DATA_ROOT}" REPO="${REPO}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

repo = Path(os.environ["REPO"])
train_json = repo / "SDPO/datasets/math_probs/train.json"
train_rows = sum(1 for line in train_json.open(encoding="utf-8") if line.strip())
if train_rows != 758:
    raise RuntimeError(f"expected 758 native math training rows, found {train_rows}: {train_json}")
print(f"training_data_rows={train_rows} path={train_json}")

root = Path(os.environ["MATH_EVAL_DATA_ROOT"])
expected = {"aime24": 30, "aime25": 30, "hmmt25": 30, "amc23": 40, "minerva": 272}
for name, expected_rows in expected.items():
    candidates = [root / name / "test.parquet", root / name / "test.jsonl", root / name / "test.json"]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing local evaluation dataset: {name} under {root}")
    if path.suffix == ".parquet":
        rows = pq.ParquetFile(path).metadata.num_rows
    elif path.suffix == ".jsonl":
        rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = len(payload if isinstance(payload, list) else payload.get("data", []))
    if rows != expected_rows:
        raise RuntimeError(f"{name}: expected {expected_rows} rows, found {rows} in {path}")
    print(f"dataset={name} rows={rows} path={path}")
PY
}

launch_phase() {
  local stop_after_step="$1"
  echo "Starting ${METHOD_LABEL} VERL training phase through step ${stop_after_step}"
  set +e
  env \
    PROJECT_ROOT="${REPO}/SDPO" \
    MODEL_PATH="${MODEL_PATH}" \
    DATA_DIR="${REPO}/SDPO/datasets/math_probs" \
    OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT}" \
    RUN_NAME="${RUN_NAME}" \
    NUM_GPUS=8 \
    SEED=0 \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" \
    ROLLOUT_N="${ROLLOUT_N}" \
    TOTAL_STEPS="${SCHEDULER_HORIZON_STEPS}" \
    STOP_AFTER_STEP="${stop_after_step}" \
    TEST_FREQ=-1 \
    SAVE_FREQ=5 \
    MAX_ACTOR_CKPT_TO_KEEP=2 \
    CHECKPOINT_SAVE_CONTENTS='[model,optimizer,extra]' \
    CHECKPOINT_LOAD_CONTENTS='[model,optimizer,extra]' \
    LEARNING_RATE=5e-6 \
    WARMUP_STEPS=0 \
    LR_SCHEDULER_TYPE=linear \
    WEIGHT_DECAY=0 \
    GRAD_CLIP=0.1 \
    ENTROPY_COEFF=1e-5 \
    TEACHER_UPDATE_RATE=0.05 \
    DIVERGENCE_ALPHA="${DIVERGENCE_ALPHA}" \
    RENYI_ORDER="${RENYI_ORDER}" \
    SELF_REFERENCE_WEIGHT="${SELF_REFERENCE_WEIGHT}" \
    REF_SYNC_STEPS=0 \
    MAX_PROMPT_LENGTH=2048 \
    MAX_RESPONSE_LENGTH=16384 \
    MAX_REPROMPT_LENGTH=16384 \
    ROLLOUT_TEMPERATURE=0.7 \
    ROLLOUT_TOP_P=0.95 \
    ROLLOUT_TOP_K=20 \
    VAL_ROLLOUT_N=1 \
    VAL_TEMPERATURE=0.7 \
    VAL_TOP_P=0.95 \
    VAL_TOP_K=20 \
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}" \
    DISTILLATION_TOPK=100 \
    DISTILLATION_ADD_TAIL=False \
    DISTILLATION_IS_CLIP=null \
    TOKEN_LOSS_CLIP=0.05 \
    LORA_RANK=64 \
    LORA_ALPHA=128 \
    TRAINER_LOGGER='[console,file]' \
    bash "${REPO}/SDPO/run_local_math_verl.sh" "${METHOD}" eval5-n16-h200 \
    2>&1 | tee -a "${LOG_ROOT}/training.log"
  local status=${PIPESTATUS[0]}
  set -e
  (( status == 0 )) || return "${status}"
  checkpoint_ready "${stop_after_step}" || {
    echo "ERROR: phase exited without a resumable native checkpoint at step ${stop_after_step}" >&2
    return 4
  }
}

evaluate_step() (
  set -Eeuo pipefail
  local step="$1"
  local actor_dir="$(checkpoint_dir "${step}")/actor"
  local merged_dir="${MERGED_ROOT}/checkpoint-${step}"
  local result_dir="${RESULT_ROOT}/checkpoint-${step}"
  rm -rf -- "${merged_dir}"
  mkdir -p "${result_dir}"
  trap 'rm -rf -- "${merged_dir}"' EXIT

  echo "Merging ${METHOD_LABEL} FSDP checkpoint-${step} for distIL evaluation"
  "${PYTHON_BIN}" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${actor_dir}" \
    --target_dir "${merged_dir}"
  test -s "${merged_dir}/config.json"

  "${PYTHON_BIN}" - "${MODEL_PATH}" "${merged_dir}" <<'PY'
import sys
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

base_dir, merged_dir = sys.argv[1:]
base_config = AutoConfig.from_pretrained(base_dir, local_files_only=True, trust_remote_code=True)
merged_config = AutoConfig.from_pretrained(merged_dir, local_files_only=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    base_dir,
    local_files_only=True,
    trust_remote_code=True,
    use_fast=True,
)
if base_config.eos_token_id != merged_config.eos_token_id:
    raise RuntimeError(
        f"merged checkpoint EOS mismatch: base={base_config.eos_token_id}, merged={merged_config.eos_token_id}"
    )
if tokenizer.eos_token_id != base_config.eos_token_id:
    raise RuntimeError(
        f"tokenizer EOS mismatch: tokenizer={tokenizer.eos_token_id}, model={base_config.eos_token_id}"
    )
if not tokenizer.is_fast:
    raise RuntimeError(f"evaluation tokenizer is not fast: {Path(base_dir)}")
print(f"Merged checkpoint/tokenizer preflight: PASS (eos={tokenizer.eos_token_id}, fast={tokenizer.is_fast})")
PY

  local lora_adapter_dir=""
  if [[ -s "${merged_dir}/lora_adapter/adapter_config.json" ]]; then
    lora_adapter_dir="${merged_dir}/lora_adapter"
    "${PYTHON_BIN}" - "${lora_adapter_dir}/adapter_config.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    config = json.load(handle)
config["lora_alpha"] = 128
with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
  fi

  MODEL_DIR="${merged_dir}" \
  TOKENIZER_DIR="${MODEL_PATH}" \
  LORA_ADAPTER_DIR="${lora_adapter_dir}" \
  MODEL_SIZE="${MODEL_SIZE}" \
  OUTPUT_DIR="${result_dir}" \
  VAL_N="${VAL_N}" \
  TENSOR_PARALLEL_SIZE=8 \
  EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.90}" \
  EVAL_SUBMISSION_MODE="${EVAL_SUBMISSION_MODE}" \
  EVAL_PROMPT_BATCH_SIZE="${EVAL_PROMPT_BATCH_SIZE}" \
  EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS}" \
  MATH_EVAL_DATA_ROOT="${MATH_EVAL_DATA_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
    bash "${REPO}/scripts/math/eval_sr_opsd_verl_math.sh"
  result_complete "${step}"
)

remove_older_native_checkpoints() {
  local current_step="$1"
  local dir step
  for dir in "${RUN_DIR}"/global_step_*; do
    [[ -d "${dir}" ]] || continue
    step="${dir##*_}"
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if (( step < current_step )); then
      remove_native_checkpoint "${step}"
    fi
  done
}

echo "============================================================"
echo "Qwen3-${MODEL_SIZE^^} ${METHOD_LABEL} native VERL train/evaluate pipeline"
echo "host=$(hostname)"
echo "hardware=${HARDWARE}"
echo "framework=SDPO native VERL"
echo "objective=${METHOD_DESCRIPTION}"
echo "shared_training=trainbs8; mbs8; rolloutn8; lr5e-6; linear/0; tok16384"
echo "training=100 physical steps; learning-rate horizon=420; external evaluation every 5 steps"
echo "evaluation=five math datasets; thinking; N=16; TP=8; max_new_tokens=${EVAL_MAX_NEW_TOKENS}; submission=${EVAL_SUBMISSION_MODE}"
echo "checkpoint_policy=retain current resume point only; delete after next checkpoint; delete final after eval"
echo "output=${RUN_ROOT}"
echo "online_loggers=disabled"
echo "============================================================"

runtime_preflight
start_adaptive_gpu_keepalive
if [[ -f "${STATE_ROOT}/complete" ]]; then
  echo "SKIP: pipeline is already complete: ${RUN_ROOT}"
  exit 0
fi

for step in $(seq 5 5 "${MAX_STEPS}"); do
  if result_complete "${step}"; then
    echo "SKIP complete evaluation at checkpoint-${step}"
  else
    if checkpoint_ready "${step}"; then
      echo "Reusing resumable native checkpoint at step ${step}"
    else
      launch_phase "${step}"
      "${ENV_DIR}/bin/ray" stop --force >/dev/null 2>&1 || true
      wait_for_gpu_release 240
    fi
    remove_older_native_checkpoints "${step}"
    evaluate_step "${step}"
    wait_for_gpu_release 240
  fi

  if checkpoint_ready "${step}"; then
    remove_older_native_checkpoints "${step}"
  fi
done

result_complete 100
remove_native_checkpoint 100
rm -f "${RUN_DIR}/latest_checkpointed_iteration.txt"
touch "${STATE_ROOT}/complete"
echo "COMPLETE: Qwen3-${MODEL_SIZE^^} ${METHOD_LABEL} trained through step 100 and all 20 checkpoints were evaluated at N=16"
echo "results=${RESULT_ROOT}"
