#!/usr/bin/env bash

ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-${ROOT}/code/distIL-sr-opsd-renyi}"
BASELINE_REPO="${BASELINE_REPO:-${ROOT}/code/distIL}"
BASELINE_OPSD="${BASELINE_REPO}/OPSD"
ENV_DIR="${ENV_DIR:-/media/vlm-ckp-fileset/ylong/sdpo/envs/opsd-math}"
CORE_RUNTIME="${CORE_RUNTIME:-/media/vlm-ckp-fileset/ylong/sdpo/runtime/math-core-torch2.8-ray2.50.1-v1}"
PYTHON_EXTRAS="${PYTHON_EXTRAS:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_assets/math-python-extras-v1}"
PYTHON_COMPLETE="${PYTHON_COMPLETE:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_assets/math-python-complete-v2}"
VLLM_COMPLETE="${VLLM_COMPLETE:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_assets/math-vllm-python-complete-v1}"
TORCH_SHM_MANAGER_ASSET="${TORCH_SHM_MANAGER_ASSET:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_assets/torch2.8/torch_shm_manager.compat}"
TORCH_HEADER_ROOT="${TORCH_HEADER_ROOT:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_assets/torch-2.8.0-headers-v1/torch/include}"
TORCH_CXX_HEADER_ROOT="${TORCH_CXX_HEADER_ROOT:-${TORCH_HEADER_ROOT}/torch/csrc/api/include}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_extensions/math-torch2.8-cu128-h200-v1}"
FLASH_SOURCE="${FLASH_SOURCE:-/media/vlm-ckp-fileset/ylong/sdpo/build/flash-attn-sm90/src}"
CONDA_ROOT="${CONDA_ROOT:-/media/damoxing/che-liu-fileset/conda}"
SITE_PACKAGES="${ENV_DIR}/lib/python3.11/site-packages"
RUNTIME_OVERLAY="${RUNTIME_OVERLAY:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_overlays/math_train_eval5_n16_h200}"
FLASH_PACKAGE_OVERLAY="${RUNTIME_OVERLAY}/flash_attn_2_8_3"
DEPENDENCY_REPAIR_OVERLAY="${MATH_DEPENDENCY_REPAIR_OVERLAY:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_overlays/math_dependency_repair_20260816}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-8B}"
MATH_TRAIN_DATA="${MATH_TRAIN_DATA:-${BASELINE_OPSD}/data/math/train.jsonl}"
MATH_EVAL_DATA_ROOT="${MATH_EVAL_DATA_ROOT:-${ROOT}/data/math_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/math_train_eval5_n16_h200_20260812}"

export ROOT REPO BASELINE_REPO BASELINE_OPSD ENV_DIR CORE_RUNTIME
export DEPENDENCY_REPAIR_OVERLAY
export PYTHON_EXTRAS PYTHON_COMPLETE VLLM_COMPLETE TORCH_SHM_MANAGER_ASSET
export TORCH_HEADER_ROOT TORCH_CXX_HEADER_ROOT TORCH_EXTENSIONS_DIR
export FLASH_SOURCE CONDA_ROOT SITE_PACKAGES RUNTIME_OVERLAY FLASH_PACKAGE_OVERLAY
export BASE_MODEL_DIR MATH_TRAIN_DATA MATH_EVAL_DATA_ROOT OUTPUT_ROOT
export SETUPTOOLS_USE_DISTUTILS=stdlib

require_file() {
  [[ -f "$1" ]] || { echo "ERROR: missing required file: $1" >&2; exit 2; }
}

require_executable() {
  [[ -x "$1" ]] || { echo "ERROR: missing executable: $1" >&2; exit 2; }
}

require_executable "${ENV_DIR}/bin/python"
require_executable "${ENV_DIR}/bin/accelerate"
require_file "${BASELINE_OPSD}/accelerate.yaml"
require_file "${BASELINE_OPSD}/opsd_train.py"
require_file "${REPO}/scripts/math/eval_sr_opsd_verl_math.sh"
require_file "${REPO}/scripts/math/validate_math_eval.py"
require_file "${REPO}/OPSD/eval/evaluate_math.py"
require_file "${BASE_MODEL_DIR}/config.json"
require_file "${MATH_TRAIN_DATA}"
require_file "${CORE_RUNTIME}/ray/dag/compiled_dag_node.py"
require_file "${TORCH_SHM_MANAGER_ASSET}"
require_file "${TORCH_HEADER_ROOT}/torch/extension.h"
require_file "${TORCH_CXX_HEADER_ROOT}/torch/all.h"
require_file "${FLASH_SOURCE}/flash_attn/__init__.py"
require_file "${SITE_PACKAGES}/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so"
require_file "${PYTHON_COMPLETE}/.complete"
require_file "${VLLM_COMPLETE}/.complete"
require_file "${VLLM_COMPLETE}/vllm/_C.abi3.so"

mkdir -p "${FLASH_PACKAGE_OVERLAY}"
if [[ -e "${FLASH_PACKAGE_OVERLAY}/flash_attn" && ! -L "${FLASH_PACKAGE_OVERLAY}/flash_attn" ]]; then
  echo "ERROR: FlashAttention overlay exists and is not a symlink" >&2
  exit 2
fi
ln -sfn "${FLASH_SOURCE}/flash_attn" "${FLASH_PACKAGE_OVERLAY}/flash_attn"

export PATH="${ENV_DIR}/bin:/usr/local/cuda/bin:/usr/bin:/bin:${PATH:-}"
export PYTHONHOME="${CONDA_ROOT}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${CORE_RUNTIME}/torch/lib:${ENV_DIR}/lib:${SITE_PACKAGES}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/media/vlm-ckp-fileset/ylong/sdpo/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/media/vlm-ckp-fileset/ylong/sdpo/cache/datasets}"
export VLLM_USE_MODELSCOPE=true
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS="${MAX_JOBS:-2}"
export CPLUS_INCLUDE_PATH="${TORCH_HEADER_ROOT}:${TORCH_CXX_HEADER_ROOT}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export CPATH="${TORCH_HEADER_ROOT}:${TORCH_CXX_HEADER_ROOT}${CPATH:+:${CPATH}}"
export SDPO_LOCAL_ONLY=1
export SWANLAB_MODE=offline
export SDPO_SWANLAB_MODE=offline
export SWANLAB_DISABLED=1
export SDPO_DEFER_SWANLAB_UPLOAD=true
export WANDB_MODE=offline
unset SWANLAB_API_KEY SWANLAB_WORKSPACE SWANLAB_PROJECT
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${HOME}/.triton/autotune"

for script in "${BASELINE_OPSD}/opsd_train.py"; do
  grep -q "selected_checkpoint_steps" "${script}" || {
    echo "ERROR: baseline runner lacks selected_checkpoint_steps: ${script}" >&2
    exit 2
  }
  grep -q "stop_after_step" "${script}" || {
    echo "ERROR: baseline runner lacks stop_after_step: ${script}" >&2
    exit 2
  }
done

repair_torch_shm_manager() {
  local target="${CORE_RUNTIME}/torch/bin/torch_shm_manager"
  if [[ ! -x "${target}" ]] || ! cmp -s "${TORCH_SHM_MANAGER_ASSET}" "${target}"; then
    mkdir -p "$(dirname "${target}")"
    local temporary="${target}.repair.$$"
    install -m 0555 "${TORCH_SHM_MANAGER_ASSET}" "${temporary}"
    mv -f "${temporary}" "${target}"
    echo "Restored torch_shm_manager: ${target}"
  fi
}

distil_pythonpath() {
  printf '%s' "${DEPENDENCY_REPAIR_OVERLAY}:${FLASH_PACKAGE_OVERLAY}:${BASELINE_OPSD}:${BASELINE_REPO}:${PYTHON_EXTRAS}:${VLLM_COMPLETE}:${PYTHON_COMPLETE}:${CORE_RUNTIME}:${SITE_PACKAGES}"
}

eval_pythonpath() {
  printf '%s' "${DEPENDENCY_REPAIR_OVERLAY}:${FLASH_PACKAGE_OVERLAY}:${REPO}/OPSD:${REPO}:${PYTHON_EXTRAS}:${VLLM_COMPLETE}:${PYTHON_COMPLETE}:${CORE_RUNTIME}:${SITE_PACKAGES}"
}

ensure_deepspeed_cpu_adam() {
  local lock_file="${TORCH_EXTENSIONS_DIR}/.cpu_adam.lock"
  command -v flock >/dev/null 2>&1 || {
    echo "ERROR: flock is required for the DeepSpeed CPUAdam prebuild" >&2
    return 1
  }

  echo "============================================================"
  echo "DeepSpeed CPUAdam preflight"
  echo "torch_headers=${TORCH_HEADER_ROOT}"
  echo "torch_extensions=${TORCH_EXTENSIONS_DIR}"
  echo "max_jobs=${MAX_JOBS}"
  echo "============================================================"
  (
    flock -x 9
    PYTHONPATH="$(distil_pythonpath)" "${ENV_DIR}/bin/python" - <<'PY'
from pathlib import Path

import torch
from deepspeed.ops.op_builder import CPUAdamBuilder

module = CPUAdamBuilder().load(verbose=True)
module_path = Path(module.__file__).resolve()
assert module_path.is_file(), module_path
for symbol in ("create_adam", "adam_update", "destroy_adam"):
    assert hasattr(module, symbol), (module_path, symbol)
print(f"torch={torch.__version__}")
print(f"cpu_adam={module_path}")
print("DeepSpeed CPUAdam build/import: PASS")
PY
  ) 9>"${lock_file}"
}

runtime_preflight() {
  repair_torch_shm_manager
  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  [[ "${gpu_count}" == "8" ]] || { echo "ERROR: expected 8 GPUs, found ${gpu_count}" >&2; exit 2; }
  if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Ev 'H200|H20Z' >/dev/null; then
    echo "ERROR: this launcher requires eight H200/H20Z GPUs" >&2
    nvidia-smi --query-gpu=index,name --format=csv,noheader >&2
    exit 2
  fi

  nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
  PYTHONPATH="$(eval_pythonpath)" "${ENV_DIR}/bin/python" - <<'PY'
import importlib
import os
from pathlib import Path

import flash_attn
import torch
from flash_attn import flash_attn_func

assert torch.cuda.device_count() == 8, torch.cuda.device_count()
assert Path(os.environ["CORE_RUNTIME"]) in Path(torch.__file__).resolve().parents, torch.__file__
for index in range(8):
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    assert capability >= (9, 0), (index, name, capability)
    print(f"cuda:{index}: {name}, capability={capability}")
for module in ("accelerate.utils.dataclasses", "datasets", "deepspeed.runtime.engine", "math_verify", "ray.dag.compiled_dag_node", "transformers", "trl", "vllm"):
    imported = importlib.import_module(module)
    print(f"{module}: {Path(imported.__file__).resolve()}")
assert flash_attn.__version__ == "2.8.3", flash_attn.__version__
manager = Path(os.environ["CORE_RUNTIME"]) / "torch/bin/torch_shm_manager"
assert manager.is_file() and os.access(manager, os.X_OK), manager
q = torch.randn((1, 16, 4, 64), device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, q, q, causal=True)
torch.cuda.synchronize()
assert out.shape == q.shape
print(f"torch={torch.__version__}")
print("SM90 BF16 FlashAttention smoke test: PASS")
PY
  ensure_deepspeed_cpu_adam
}

validate_inputs() {
  local rows
  rows="$(wc -l < "${MATH_TRAIN_DATA}" | tr -d '[:space:]')"
  [[ "${rows}" == "758" ]] || {
    echo "ERROR: expected 758 math training rows, found ${rows}: ${MATH_TRAIN_DATA}" >&2
    exit 2
  }
  PYTHONPATH="$(eval_pythonpath)" "${ENV_DIR}/bin/python" - <<'PY'
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

wait_for_gpu_release() {
  local timeout_seconds="${1:-180}"
  local deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '^[0-9]+'; then
      sleep 5
    else
      return 0
    fi
  done
  echo "ERROR: GPU processes did not exit within ${timeout_seconds}s" >&2
  nvidia-smi >&2 || true
  return 1
}
