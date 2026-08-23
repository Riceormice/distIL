#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-${ROOT}/code/distIL-sr-opsd-renyi}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${ROOT}/code/SDPO-main-latest-provided}"
PYTHON_ENV="${PYTHON_ENV:-/media/damoxing/che-liu-fileset/ylong/sdpo/envs/math-verl-current}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_ENV}/bin/python}"
MODEL_PATH="${MODEL_PATH:-/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sdpo_physics_rho_selfref_grid_eval5_nockpt}"
EXPERIMENT_SUFFIX="${EXPERIMENT_SUFFIX:-rho-selfref-grid-v1}"
RUNNER="${SCRIPT_DIR}/run_physics_grid_point_a800.sh"
COLLECTOR="${SCRIPT_DIR}/collect_physics_eval5_metrics.py"

SELF_REFERENCE_COEFFICIENT=0.9
RENYI_ORDER=0.9
SEED=0
TOTAL_TRAINING_STEPS=420
TEST_FREQ=5
VAL_N=16
EXPECTED_VALIDATION_LINES=1280

METHOD_LABEL=sr_opsd_forward_renyi
SELECTOR_ALPHA=0.25
ENTROPY_COEFF=1e-5
TEACHER_UPDATE_RATE=0.05
DISTILLATION_TOPK=100
IS_CLIP=2.0
TRAIN_BATCH_SIZE=32
PPO_MINI_BATCH_SIZE=32
ROLLOUT_N=8
LR=1e-5
WARMUP_STEPS=10

EXP_NAME="physics-${METHOD_LABEL}-refTrue-selectorAlpha${SELECTOR_ALPHA}-rho${RENYI_ORDER}-selfref${SELF_REFERENCE_COEFFICIENT}-sync0-entropy${ENTROPY_COEFF}-ema${TEACHER_UPDATE_RATE}-topk${DISTILLATION_TOPK}-tailTrue-fullLogitTrue-isclip${IS_CLIP}-steps${TOTAL_TRAINING_STEPS}-trainbs${TRAIN_BATCH_SIZE}-mbs${PPO_MINI_BATCH_SIZE}-rolloutn${ROLLOUT_N}-lr${LR}-warmup${WARMUP_STEPS}-seed${SEED}-modelQwen3-8B-${EXPERIMENT_SUFFIX}"
RUN_DIR="${OUTPUT_ROOT}/runs/${EXP_NAME}"
LOG_DIR="${OUTPUT_ROOT}/logs/${EXP_NAME}"
LAUNCH_DIR="${OUTPUT_ROOT}/launcher"
LAUNCH_LOG="${LAUNCH_DIR}/alpha090_rho090_a800_$(date +%Y%m%d_%H%M%S).log"
LOCK_FILE="${OUTPUT_ROOT}/alpha090_rho090_a800.lock"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_physics_alpha090_rho090_a800}"

for required in \
  "${PYTHON_BIN}" \
  "${PYTHON_ENV}/.math-env-complete" \
  "${PYTHON_ENV}/lib/python3.11/site-packages/torch/bin/torch_shm_manager" \
  "${MODEL_PATH}/config.json" \
  "${PROJECT_ROOT}/datasets/sciknoweval/physics/train.parquet" \
  "${PROJECT_ROOT}/datasets/sciknoweval/physics/test.parquet" \
  "${PROJECT_ROOT}/verl/utils/reward_score/feedback/__init__.py" \
  "${RUNNER}" \
  "${COLLECTOR}"
do
  [[ -e "${required}" ]] || { echo "ERROR: missing required path: ${required}" >&2; exit 3; }
done

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${LAUNCH_DIR}"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1
exec 9>"${LOCK_FILE}"
flock -n 9 || { echo "ERROR: alpha=0.90 rho=0.90 pipeline is already running" >&2; exit 4; }

export PATH="${PYTHON_ENV}/bin:/usr/local/cuda/bin:/usr/bin:/bin:${PATH:-}"
export VIRTUAL_ENV="${PYTHON_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}"
export LD_LIBRARY_PATH="${PYTHON_ENV}/lib:${PYTHON_ENV}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export SETUPTOOLS_USE_DISTUTILS=stdlib
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS_PER_NODE=8
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export RAY_DEDUP_LOGS=0
export VLLM_USE_MODELSCOPE=true
export VLLM_USE_V1=1
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_CUDA_ARCH_LIST=8.0
export EXPECTED_CUDA_CAPABILITY=8.0
export REF_SYNC_STEPS=0
export SAVE_FREQ=-1
export SWANLAB_MODE=disabled
export SDPO_SWANLAB_MODE=disabled
export SWANLAB_DISABLED=1
export SDPO_LOCAL_ONLY=1
unset PYTHONHOME PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
unset SWANLAB_API_KEY SWANLAB_WORKSPACE SWANLAB_PROJECT
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_MODE WANDB_NAME
ulimit -n 65535 2>/dev/null || true

cleanup_runtime() {
  "${PYTHON_ENV}/bin/ray" stop --force >/dev/null 2>&1 || true
  rm -rf -- "${RAY_TEMP_DIR}" 2>/dev/null || true
}
trap cleanup_runtime EXIT

validate_complete() {
  [[ -s "${LOG_DIR}/metrics.jsonl" && -d "${LOG_DIR}/validation" ]] || return 1
  if find "${RUN_DIR}" -mindepth 1 -type d -name 'global_step_*' -print -quit 2>/dev/null | grep -q .; then
    echo "ERROR: checkpoint found although checkpoint saving is disabled" >&2
    return 1
  fi
  "${PYTHON_BIN}" "${COLLECTOR}" \
    --metrics-jsonl "${LOG_DIR}/metrics.jsonl" \
    --validation-dir "${LOG_DIR}/validation" \
    --output-csv "${LOG_DIR}/eval5_metrics.csv" \
    --total-steps "${TOTAL_TRAINING_STEPS}" \
    --eval-freq "${TEST_FREQ}" \
    --expected-validation-lines "${EXPECTED_VALIDATION_LINES}"
}

echo "============================================================"
echo "Qwen3-8B Physics missing grid point"
echo "self_reference_alpha=${SELF_REFERENCE_COEFFICIENT}"
echo "renyi_rho=${RENYI_ORDER}"
echo "selector_alpha=${SELECTOR_ALPHA}"
echo "seed=${SEED}"
echo "training=${TOTAL_TRAINING_STEPS} steps"
echo "evaluation=every ${TEST_FREQ} steps, N=${VAL_N}"
echo "reference_sync=0"
echo "checkpoints=disabled"
echo "output=${OUTPUT_ROOT}"
echo "launch_log=${LAUNCH_LOG}"
echo "============================================================"

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
[[ "${GPU_COUNT}" == "8" ]] || { echo "ERROR: expected 8 GPUs, found ${GPU_COUNT}" >&2; exit 2; }
if nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | grep -Ev 'A800.*8\.0' >/dev/null; then
  echo "ERROR: this launcher requires eight NVIDIA A800 SM80 GPUs" >&2
  nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader

"${PYTHON_BIN}" - <<'PY'
import importlib
import torch
from flash_attn import flash_attn_func

assert torch.cuda.device_count() == 8, torch.cuda.device_count()
for index in range(8):
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    if "A800" not in name or capability != (8, 0):
        raise RuntimeError((index, name, capability))
for module_name in ("ray", "vllm", "transformers", "verl"):
    module = importlib.import_module(module_name)
    print(f"{module_name}={getattr(module, '__file__', '<namespace>')}")
q = torch.randn((1, 32, 4, 64), device="cuda:0", dtype=torch.bfloat16)
result = flash_attn_func(q, q, q, causal=True)
torch.cuda.synchronize()
assert result.shape == q.shape
print("A800 runtime preflight: PASS")
PY

if validate_complete; then
  touch "${RUN_DIR}/TRAINING_COMPLETE"
  echo "SKIP: alpha=0.90 rho=0.90 is already complete"
  exit 0
fi

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "PREFLIGHT_ONLY=1: training was not started"
  exit 0
fi

cleanup_runtime
mkdir -p "${RAY_TEMP_DIR}"
rm -f -- "${RUN_DIR}/TRAINING_COMPLETE"
set +e
SELF_REFERENCE_COEFFICIENT="${SELF_REFERENCE_COEFFICIENT}" \
  SEED="${SEED}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  EXPERIMENT_SUFFIX="${EXPERIMENT_SUFFIX}" \
  TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
  TEST_FREQ="${TEST_FREQ}" \
  VAL_N="${VAL_N}" \
  SAVE_FREQ=-1 \
  RAY_TEMP_DIR="${RAY_TEMP_DIR}" \
  bash "${RUNNER}" sr_opsd_ref "${RENYI_ORDER}"
TRAIN_EXIT=$?
set -e

if validate_complete; then
  touch "${RUN_DIR}/TRAINING_COMPLETE"
  if (( TRAIN_EXIT == 0 )); then
    echo "COMPLETE: alpha=0.90 rho=0.90"
  else
    echo "RECOVERED COMPLETE: trainer exited ${TRAIN_EXIT}, but all metrics are complete"
  fi
  exit 0
fi

echo "ERROR: alpha=0.90 rho=0.90 is incomplete; trainer exit=${TRAIN_EXIT}" >&2
(( TRAIN_EXIT != 0 )) && exit "${TRAIN_EXIT}"
exit 5
