#!/usr/bin/env bash
set -Eeuo pipefail

METHOD="${1:?Usage: $0 grpo|sdpo|sr_opsd}"
case "${METHOD}" in
  grpo|sdpo|sr_opsd) ;;
  *) echo "ERROR: METHOD must be grpo, sdpo, or sr_opsd" >&2; exit 2 ;;
esac

ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-${ROOT}/code/distIL-sr-opsd-renyi}"
ENV_DIR="${ENV_DIR:-/media/vlm-ckp-fileset/ylong/sdpo/envs/verl-vllm010-h200-v2}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"
DEPENDENCY_REPAIR_OVERLAY="${MATH_DEPENDENCY_REPAIR_OVERLAY:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_overlays/math_dependency_repair_20260816}"
TORCH_SHM_MANAGER_ASSET="${TORCH_SHM_MANAGER_ASSET:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_assets/torch2.8/torch_shm_manager.compat}"
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

VAL_N=16
MAX_STEPS=100
SCHEDULER_HORIZON_STEPS=420
TRAIN_BATCH_SIZE=8
PPO_MINI_BATCH_SIZE=8
ROLLOUT_N=8

case "${METHOD}" in
  grpo)
    METHOD_LABEL=GRPO
    METHOD_DIR=grpo
    RUN_NAME="grpo-${MODEL_SIZE}-seed0-native-verl-lr5e-6-trainbs8-mbs8-rolloutn8-eps0.2-temp0.7-tok16384-steps100-sched420-eval5-n16-${HARDWARE}"
    METHOD_DESCRIPTION="GRPO; epsilon=0.2; group-normalized advantages"
    ;;
  sdpo)
    METHOD_LABEL="SDPO (RKL)"
    METHOD_DIR=sdpo
    RUN_NAME="sdpo-${MODEL_SIZE}-seed0-native-verl-rkl-ema0.05-lr5e-6-trainbs8-mbs8-rolloutn8-topk100-tailFalse-clip0.05-temp0.7-tok16384-steps100-sched420-eval5-n16-${HARDWARE}"
    METHOD_DESCRIPTION="Reverse KL; EMA teacher=0.05; no frozen-reference anchoring"
    ;;
  sr_opsd)
    METHOD_LABEL=SR-OPSD
    METHOD_DIR=sr_opsd
    RUN_NAME="sr-opsd-${MODEL_SIZE}-seed0-native-verl-forward-renyi-rho0.95-refw0.9-sync0-ema0.05-lr5e-6-trainbs8-mbs8-rolloutn8-topk100-tailFalse-clip0.05-temp0.7-tok16384-steps100-sched420-eval5-n16-${HARDWARE}"
    METHOD_DESCRIPTION="Forward Renyi rho=0.95; self-reference=0.9; frozen reference sync=0"
    ;;
esac

RUN_ROOT="${OUTPUT_ROOT}/${METHOD_DIR}/${RUN_NAME}"
TRAIN_OUTPUT_ROOT="${RUN_ROOT}/native"
RUN_DIR="${TRAIN_OUTPUT_ROOT}/checkpoints/${RUN_NAME}"
RESULT_ROOT="${RUN_ROOT}/evaluations"
MERGED_ROOT="${RUN_ROOT}/merged"
LOG_ROOT="${RUN_ROOT}/logs"
STATE_ROOT="${RUN_ROOT}/state"

unset PYTHONHOME CONDA_PREFIX
export PATH="${ENV_DIR}/bin:/usr/local/cuda/bin:/usr/bin:/bin:${PATH:-}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "${DEPENDENCY_REPAIR_OVERLAY}/.complete" ]]; then
  export PYTHONPATH="${REPO}/SDPO:${REPO}:${DEPENDENCY_REPAIR_OVERLAY}"
else
  export PYTHONPATH="${REPO}/SDPO:${REPO}"
fi
export PYTHON_BIN
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export SETUPTOOLS_USE_DISTUTILS=stdlib
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

test -x "${PYTHON_BIN}"
test -f "${TORCH_SHM_MANAGER_ASSET}"
test -f "${MODEL_PATH}/config.json"
test -f "${REPO}/SDPO/datasets/math_probs/train.json"
test -f "${REPO}/SDPO/datasets/math_probs/test.json"
test -f "${REPO}/SDPO/run_local_math_verl.sh"
test -f "${REPO}/scripts/math/eval_sr_opsd_verl_math.sh"
test -f "${REPO}/scripts/math/validate_math_eval.py"
mkdir -p "${RUN_DIR}" "${RESULT_ROOT}" "${MERGED_ROOT}" "${LOG_ROOT}" "${STATE_ROOT}"

exec 9>"${STATE_ROOT}/pipeline.lock"
flock -n 9 || { echo "ERROR: ${METHOD_LABEL} pipeline is already running: ${RUN_ROOT}" >&2; exit 3; }
LAUNCH_LOG="${LOG_ROOT}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

cat >"${STATE_ROOT}/parameters.env" <<EOF
method=${METHOD}
framework=SDPO-native-VERL
model_size=${MODEL_SIZE}
hardware=${HARDWARE}
model=${MODEL_PATH}
seed=0
train_batch_size=${TRAIN_BATCH_SIZE}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
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
  while (( SECONDS < deadline )); do
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '^[0-9]+'; then
      sleep 5
    else
      return 0
    fi
  done
  echo "ERROR: GPU processes did not exit in time" >&2
  nvidia-smi >&2 || true
  return 1
}

repair_torch_shm_manager() {
  local target="${ENV_DIR}/lib/python3.11/site-packages/torch/bin/torch_shm_manager"
  if [[ ! -x "${target}" ]] || ! cmp -s "${TORCH_SHM_MANAGER_ASSET}" "${target}"; then
    mkdir -p "$(dirname "${target}")"
    local temporary="${target}.repair.$$"
    install -m 0555 "${TORCH_SHM_MANAGER_ASSET}" "${temporary}"
    mv -f "${temporary}" "${target}"
    echo "Restored native VERL torch_shm_manager: ${target}"
  fi
}

runtime_preflight() {
  repair_torch_shm_manager
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
    DIVERGENCE_ALPHA=0.25 \
    RENYI_ORDER=0.95 \
    SELF_REFERENCE_WEIGHT=0.9 \
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
  LORA_ADAPTER_DIR="${lora_adapter_dir}" \
  MODEL_SIZE="${MODEL_SIZE}" \
  OUTPUT_DIR="${result_dir}" \
  VAL_N="${VAL_N}" \
  TENSOR_PARALLEL_SIZE=8 \
  EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.90}" \
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
echo "evaluation=five math datasets; thinking; N=16; TP=8"
echo "checkpoint_policy=retain current resume point only; delete after next checkpoint; delete final after eval"
echo "output=${RUN_ROOT}"
echo "online_loggers=disabled"
echo "============================================================"

runtime_preflight
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
