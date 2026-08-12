#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-${ROOT}/code/distIL-sr-opsd-renyi}"
ENV_DIR="${ENV_DIR:-/media/vlm-ckp-fileset/ylong/sdpo/envs/opsd-math}"
CORE_RUNTIME="${CORE_RUNTIME:-/media/vlm-ckp-fileset/ylong/sdpo/runtime/math-core-torch2.8-ray2.50.1-v1}"
TORCH_SHM_MANAGER_ASSET="${TORCH_SHM_MANAGER_ASSET:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_assets/torch2.8/torch_shm_manager.compat}"
PYTHON_EXTRAS="${PYTHON_EXTRAS:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_assets/math-python-extras-v1}"
FLASH_SOURCE="${FLASH_SOURCE:-/media/vlm-ckp-fileset/ylong/sdpo/build/flash-attn-sm90/src}"
CONDA_ROOT="${CONDA_ROOT:-/media/damoxing/che-liu-fileset/conda}"
SITE_PACKAGES="${ENV_DIR}/lib/python3.11/site-packages"
RUNTIME_OVERLAY="${RUNTIME_OVERLAY:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_overlays/math_eval_n12_n16_a800}"
FLASH_PACKAGE_OVERLAY="${RUNTIME_OVERLAY}/flash_attn_2_8_3"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-8B}"
MATH_EVAL_DATA_ROOT="${MATH_EVAL_DATA_ROOT:-${ROOT}/data/math_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/math_eval_n12_n16_20260812}"

export ROOT REPO ENV_DIR CORE_RUNTIME TORCH_SHM_MANAGER_ASSET PYTHON_EXTRAS
export FLASH_SOURCE CONDA_ROOT SITE_PACKAGES RUNTIME_OVERLAY FLASH_PACKAGE_OVERLAY
export BASE_MODEL_DIR MATH_EVAL_DATA_ROOT OUTPUT_ROOT

require_file() {
  [[ -f "$1" ]] || { echo "ERROR: missing required file: $1" >&2; exit 2; }
}

require_executable() {
  [[ -x "$1" ]] || { echo "ERROR: missing executable: $1" >&2; exit 2; }
}

require_executable "${ENV_DIR}/bin/python"
require_file "${REPO}/scripts/math/eval_sr_opsd_verl_math.sh"
require_file "${REPO}/scripts/math/validate_math_eval.py"
require_file "${REPO}/OPSD/eval/evaluate_math.py"
require_file "${BASE_MODEL_DIR}/config.json"
require_file "${CORE_RUNTIME}/ray/dag/compiled_dag_node.py"
require_file "${TORCH_SHM_MANAGER_ASSET}"
require_file "${FLASH_SOURCE}/flash_attn/__init__.py"
require_file "${SITE_PACKAGES}/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so"

mkdir -p "${FLASH_PACKAGE_OVERLAY}"
if [[ -L "${FLASH_PACKAGE_OVERLAY}/flash_attn" ]]; then
  [[ "$(readlink -f "${FLASH_PACKAGE_OVERLAY}/flash_attn")" == "$(readlink -f "${FLASH_SOURCE}/flash_attn")" ]] || {
    echo "ERROR: unexpected FlashAttention overlay target" >&2
    exit 2
  }
elif [[ -e "${FLASH_PACKAGE_OVERLAY}/flash_attn" ]]; then
  echo "ERROR: FlashAttention overlay exists and is not a symlink" >&2
  exit 2
else
  ln -s "${FLASH_SOURCE}/flash_attn" "${FLASH_PACKAGE_OVERLAY}/flash_attn"
fi

export PATH="${ENV_DIR}/bin:/usr/local/cuda/bin:/usr/bin:/bin:${PATH:-}"
export PYTHONHOME="${CONDA_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${FLASH_PACKAGE_OVERLAY}:${REPO}/OPSD:${REPO}/SDPO:${REPO}:${PYTHON_EXTRAS}:${CORE_RUNTIME}:${SITE_PACKAGES}"
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
export TORCH_CUDA_ARCH_LIST=8.0
export SDPO_LOCAL_ONLY=1
export SWANLAB_MODE=offline
export SDPO_SWANLAB_MODE=offline
export SWANLAB_DISABLED=1
export WANDB_MODE=offline
unset SWANLAB_API_KEY SWANLAB_WORKSPACE SWANLAB_PROJECT
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

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

python() {
  repair_torch_shm_manager
  "${ENV_DIR}/bin/python" "$@"
}
export -f repair_torch_shm_manager python
repair_torch_shm_manager

runtime_preflight() {
  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  [[ "${gpu_count}" == "8" ]] || { echo "ERROR: expected 8 GPUs, found ${gpu_count}" >&2; exit 2; }

  nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
  if nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -Ev '^8\.0$' >/dev/null; then
    echo "ERROR: this launcher requires eight SM80 A800 GPUs" >&2
    exit 2
  fi

  "${ENV_DIR}/bin/python" - <<'PY'
import importlib
import os
from pathlib import Path

import torch

assert torch.cuda.device_count() == 8, torch.cuda.device_count()
for index in range(8):
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    assert capability == (8, 0), (index, name, capability)
    print(f"cuda:{index}: {name}, capability={capability}")

for module in ("flash_attn", "math_verify", "ray.dag.compiled_dag_node", "transformers", "vllm"):
    imported = importlib.import_module(module)
    print(f"{module}: {getattr(imported, '__file__', '<namespace>')}")

manager = Path(os.environ["CORE_RUNTIME"]) / "torch/bin/torch_shm_manager"
assert manager.is_file() and os.access(manager, os.X_OK), manager
print(f"torch={torch.__version__}")
print("A800 evaluation runtime preflight: PASS")
PY
}

validate_local_datasets() {
  "${ENV_DIR}/bin/python" - <<'PY'
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
print("Local math datasets: PASS")
PY
}
