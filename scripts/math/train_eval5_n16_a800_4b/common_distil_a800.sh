#!/usr/bin/env bash

ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-${ROOT}/code/distIL-sr-opsd-renyi}"
BASELINE_REPO="${BASELINE_REPO:-${ROOT}/code/distIL}"
BASELINE_OPSD="${BASELINE_REPO}/OPSD"
source "${REPO}/scripts/math/unified_env/activate_unified_math_env.sh" opsd
TORCH_HEADER_ROOT="${TORCH_HEADER_ROOT:-${ENV_DIR}/lib/python3.11/site-packages/torch/include}"
TORCH_CXX_HEADER_ROOT="${TORCH_CXX_HEADER_ROOT:-${TORCH_HEADER_ROOT}/torch/csrc/api/include}"
SITE_PACKAGES="${ENV_DIR}/lib/python3.11/site-packages"

if [[ -z "${BASE_MODEL_DIR:-}" ]]; then
  for candidate in \
    /media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-4B \
    /media/vlm-ckp-fileset/ylong/models/Qwen3-4B \
    /media/damoxing/che-liu-fileset/ylong/sdpo/models/Qwen3-4B
  do
    if [[ -s "${candidate}/config.json" ]]; then
      BASE_MODEL_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${BASE_MODEL_DIR:-}" ]]; then
  while IFS= read -r config_path; do
    BASE_MODEL_DIR="${config_path%/config.json}"
    break
  done < <(
    find /media/vlm-ckp-fileset/ylong /media/damoxing/che-liu-fileset/ylong \
      -maxdepth 7 -type f -path '*/Qwen3-4B/config.json' \
      -print 2>/dev/null
  )
fi
if [[ -z "${BASE_MODEL_DIR:-}" || ! -s "${BASE_MODEL_DIR}/config.json" ]]; then
  echo "ERROR: Qwen3-4B was not found." >&2
  echo "Set BASE_MODEL_DIR to its exact local directory before launching." >&2
  exit 2
fi
MATH_TRAIN_DATA="${MATH_TRAIN_DATA:-${BASELINE_OPSD}/data/math/train.jsonl}"
MATH_EVAL_DATA_ROOT="${MATH_EVAL_DATA_ROOT:-${ROOT}/data/math_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/math_4b_train_eval5_n16_a800_20260812}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_extensions/math-torch-a800-v1}"

export ROOT REPO BASELINE_REPO BASELINE_OPSD ENV_DIR
export TORCH_HEADER_ROOT TORCH_CXX_HEADER_ROOT SITE_PACKAGES
export BASE_MODEL_DIR MATH_TRAIN_DATA MATH_EVAL_DATA_ROOT OUTPUT_ROOT TORCH_EXTENSIONS_DIR
unset SETUPTOOLS_USE_DISTUTILS

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
require_file "${ENV_DIR}/.math-env-complete"
require_executable "${SITE_PACKAGES}/torch/bin/torch_shm_manager"
require_file "${TORCH_HEADER_ROOT}/torch/extension.h"
require_file "${TORCH_CXX_HEADER_ROOT}/torch/all.h"
require_file "${SITE_PACKAGES}/flash_attn/__init__.py"
compgen -G "${SITE_PACKAGES}/flash_attn_2_cuda*.so" >/dev/null
compgen -G "${SITE_PACKAGES}/vllm/_C*.so" >/dev/null

export PYTHONPATH="${BASELINE_OPSD}:${BASELINE_REPO}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/media/vlm-ckp-fileset/ylong/sdpo/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/media/vlm-ckp-fileset/ylong/sdpo/cache/datasets}"
# Models and evaluation datasets are local. Enabling ModelScope makes vLLM
# import the optional `modelscope` package before training starts.
unset VLLM_USE_MODELSCOPE
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_CUDA_ARCH_LIST=8.0
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

distil_pythonpath() {
  printf '%s' "${BASELINE_OPSD}:${BASELINE_REPO}"
}

eval_pythonpath() {
  printf '%s' "${REPO}/OPSD:${REPO}"
}

ensure_deepspeed_cpu_adam() {
  local lock_file="${TORCH_EXTENSIONS_DIR}/.cpu_adam.lock"
  command -v flock >/dev/null 2>&1 || {
    echo "ERROR: flock is required for the DeepSpeed CPUAdam prebuild" >&2
    return 1
  }
  (
    flock -x 9
    PYTHONPATH="$(distil_pythonpath)" "${ENV_DIR}/bin/python" - <<'PY'
import torch
from deepspeed.ops.op_builder import CPUAdamBuilder

module = CPUAdamBuilder().load(verbose=True)
from pathlib import Path
module_path = Path(module.__file__).resolve()
assert module_path.is_file(), module_path
for symbol in ("create_adam", "adam_update", "destroy_adam"):
    assert hasattr(module, symbol), (module_path, symbol)
print(f"DeepSpeed CPUAdam: PASS ({module_path})")
PY
  ) 9>"${lock_file}"
}

runtime_preflight() {
  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  [[ "${gpu_count}" == "8" ]] || { echo "ERROR: expected 8 GPUs, found ${gpu_count}" >&2; exit 2; }
  if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Ev 'A800' >/dev/null; then
    echo "ERROR: this launcher requires eight A800 GPUs" >&2
    nvidia-smi --query-gpu=index,name --format=csv,noheader >&2
    exit 2
  fi
  nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader

  env -u PYTHONHOME -u CONDA_PREFIX \
    PYTHONPATH="$(eval_pythonpath)" \
    "${ENV_DIR}/bin/python" "${REPO}/scripts/math/unified_env/verify_unified_math_env.py" \
      --profile opsd --prefix "${ENV_DIR}" --repo "${REPO}" --gpu-smoke

  PYTHONPATH="$(eval_pythonpath)" "${ENV_DIR}/bin/python" - <<'PY'
import importlib
import os
from pathlib import Path

import torch
from flash_attn import flash_attn_func

assert torch.cuda.device_count() == 8, torch.cuda.device_count()
assert Path(os.environ["ENV_DIR"]).resolve() in Path(torch.__file__).resolve().parents, torch.__file__
for index in range(8):
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    assert capability == (8, 0), (index, name, capability)
for name in ("accelerate.utils.dataclasses", "datasets", "deepspeed.runtime.engine", "flash_attn", "math_verify", "ray", "transformers", "trl", "vllm"):
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__file__', '<namespace>')}")
q = torch.randn((1, 16, 4, 64), device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, q, q, causal=True)
torch.cuda.synchronize()
assert out.shape == q.shape
print(f"torch={torch.__version__}")
print(f"model={os.environ['BASE_MODEL_DIR']}")
print("A800 BF16 FlashAttention smoke test: PASS")
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
