#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-${ROOT}/code/distIL-sr-opsd-renyi}"
ENV_DIR="${ENV_DIR:-/media/vlm-ckp-fileset/ylong/sdpo/envs/verl-vllm010-h200-v2}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"
MODEL_PATH="${MODEL_PATH:-/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-8B}"
MATH_EVAL_DATA_ROOT="${MATH_EVAL_DATA_ROOT:-${ROOT}/data/math_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/math_train_eval5_n16_h200_20260812}"
RUN_NAME="sr-opsd-8b-seed0-forward-renyi-rho0.95-refw0.9-sync0-lr5e-6-tok16384-steps100-sched420-eval5-n16-h200"
RUN_ROOT="${OUTPUT_ROOT}/sr_opsd/${RUN_NAME}"
TRAIN_OUTPUT_ROOT="${RUN_ROOT}/native"
RUN_DIR="${TRAIN_OUTPUT_ROOT}/checkpoints/${RUN_NAME}"
RESULT_ROOT="${RUN_ROOT}/evaluations"
MERGED_ROOT="${RUN_ROOT}/merged"
LOG_ROOT="${RUN_ROOT}/logs"
STATE_ROOT="${RUN_ROOT}/state"
VAL_N=16
MAX_STEPS=100

unset PYTHONHOME CONDA_PREFIX
export PATH="${ENV_DIR}/bin:/usr/local/cuda/bin:/usr/bin:/bin:${PATH:-}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}/SDPO:${REPO}"
export PYTHON_BIN
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
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
test -f "${MODEL_PATH}/config.json"
test -f "${REPO}/SDPO/datasets/math_probs/train.json"
test -f "${REPO}/SDPO/datasets/math_probs/test.json"
test -f "${REPO}/scripts/math/eval_sr_opsd_verl_math.sh"
test -f "${REPO}/scripts/math/validate_math_eval.py"
mkdir -p "${RUN_DIR}" "${RESULT_ROOT}" "${MERGED_ROOT}" "${LOG_ROOT}" "${STATE_ROOT}"

exec 9>"${STATE_ROOT}/pipeline.lock"
flock -n 9 || { echo "ERROR: SR-OPSD pipeline is already running: ${RUN_ROOT}" >&2; exit 3; }
LAUNCH_LOG="${LOG_ROOT}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

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

runtime_preflight() {
  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  [[ "${gpu_count}" == "8" ]] || { echo "ERROR: expected 8 GPUs, found ${gpu_count}" >&2; exit 2; }
  if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Ev 'H200|H20Z' >/dev/null; then
    echo "ERROR: this launcher requires eight H200/H20Z GPUs" >&2
    exit 2
  fi
  nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
  "${PYTHON_BIN}" - <<'PY'
import importlib
import torch

assert torch.cuda.device_count() == 8, torch.cuda.device_count()
for index in range(8):
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    assert capability >= (9, 0), (index, name, capability)
for module in ("flash_attn", "ray", "transformers", "vllm", "verl"):
    imported = importlib.import_module(module)
    print(f"{module}: {getattr(imported, '__file__', '<namespace>')}")
print(f"torch={torch.__version__}")
PY

  MATH_EVAL_DATA_ROOT="${MATH_EVAL_DATA_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["MATH_EVAL_DATA_ROOT"])
expected = {"aime24": 30, "aime25": 30, "hmmt25": 30, "amc23": 40, "minerva": 272}
for name, expected_rows in expected.items():
    candidates = [root / name / "test.parquet", root / name / "test.jsonl", root / name / "test.json"]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing local evaluation dataset: {name} under {root}")
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
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
  echo "Starting SR-OPSD training phase through step ${stop_after_step}"
  set +e
  env \
    PROJECT_ROOT="${REPO}/SDPO" \
    MODEL_PATH="${MODEL_PATH}" \
    MODEL_SIZE=8b \
    DATA_DIR="${REPO}/SDPO/datasets/math_probs" \
    OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT}" \
    RUN_NAME="${RUN_NAME}" \
    NUM_GPUS=8 \
    SEED=0 \
    TRAIN_BATCH_SIZE=8 \
    PPO_MINI_BATCH_SIZE=8 \
    ROLLOUT_N=1 \
    TOTAL_STEPS=420 \
    STOP_AFTER_STEP="${stop_after_step}" \
    TEST_FREQ=-1 \
    SAVE_FREQ=5 \
    MAX_ACTOR_CKPT_TO_KEEP=2 \
    CHECKPOINT_SAVE_CONTENTS='[model,optimizer,extra]' \
    CHECKPOINT_LOAD_CONTENTS='[model,optimizer,extra]' \
    SAVE_TEACHER_CHECKPOINT=True \
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
    LORA_RANK=0 \
    LORA_ALPHA=128 \
    TRAINER_LOGGER='[console,file]' \
    bash "${REPO}/SDPO/run_local_ours_math.sh" eval5-n16-h200 \
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

  echo "Merging SR-OPSD FSDP checkpoint-${step} for distIL evaluation"
  "${PYTHON_BIN}" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${actor_dir}" \
    --target_dir "${merged_dir}"
  test -s "${merged_dir}/config.json"

  MODEL_DIR="${merged_dir}" \
  LORA_ADAPTER_DIR="" \
  MODEL_SIZE=8b \
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
echo "Qwen3-8B SR-OPSD train/evaluate pipeline"
echo "host=$(hostname)"
echo "training=100 actual steps; learning-rate horizon=420; external distIL evaluation every 5 steps"
echo "evaluation=five math datasets; thinking; N=16; TP=8"
echo "objective=Forward Renyi rho=0.95; self-reference=0.9; frozen reference sync=0"
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
echo "COMPLETE: SR-OPSD trained through step 100 and all 20 checkpoints were evaluated at N=16"
echo "results=${RESULT_ROOT}"
