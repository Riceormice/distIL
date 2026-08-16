#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-${ROOT}/code/distIL-sr-opsd-renyi}"
source "${REPO}/scripts/math/unified_env/activate_unified_math_env.sh" opsd
SITE_PACKAGES="${ENV_DIR}/lib/python3.11/site-packages"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-8B}"
MATH_EVAL_DATA_ROOT="${MATH_EVAL_DATA_ROOT:-${ROOT}/data/math_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/math_eval_n16_h200_20260812}"

export ROOT REPO ENV_DIR SITE_PACKAGES BASE_MODEL_DIR MATH_EVAL_DATA_ROOT OUTPUT_ROOT

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
require_file "${ENV_DIR}/.math-env-complete"
require_executable "${SITE_PACKAGES}/torch/bin/torch_shm_manager"
require_file "${SITE_PACKAGES}/flash_attn/__init__.py"
compgen -G "${SITE_PACKAGES}/flash_attn_2_cuda*.so" >/dev/null
compgen -G "${SITE_PACKAGES}/vllm/_C*.so" >/dev/null

export PYTHONPATH="${REPO}/OPSD:${REPO}/SDPO:${REPO}"
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
export SDPO_LOCAL_ONLY=1
export SWANLAB_MODE=offline
export SDPO_SWANLAB_MODE=offline
export SWANLAB_DISABLED=1
export WANDB_MODE=offline
unset SWANLAB_API_KEY SWANLAB_WORKSPACE SWANLAB_PROJECT
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

python() {
  "${ENV_DIR}/bin/python" "$@"
}
export -f python

runtime_preflight() {
  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  [[ "${gpu_count}" == "8" ]] || { echo "ERROR: expected 8 GPUs, found ${gpu_count}" >&2; exit 2; }

  nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
  "${ENV_DIR}/bin/python" "${REPO}/scripts/math/unified_env/verify_unified_math_env.py" \
    --profile opsd --prefix "${ENV_DIR}" --repo "${REPO}" --gpu-smoke
  "${ENV_DIR}/bin/python" - <<'PY'
import importlib
import importlib.metadata
import os
from pathlib import Path

import flash_attn
import torch
from flash_attn import flash_attn_func

assert torch.cuda.device_count() == 8, torch.cuda.device_count()
assert Path(os.environ["ENV_DIR"]).resolve() in Path(torch.__file__).resolve().parents, torch.__file__
for index in range(8):
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    assert capability >= (9, 0), (index, name, capability)
    print(f"cuda:{index}: {name}, capability={capability}")

for module in ("datasets", "flash_attn", "math_verify", "ray.dag.compiled_dag_node", "transformers", "vllm"):
    imported = importlib.import_module(module)
    path = Path(imported.__file__).resolve()
    assert Path(os.environ["ENV_DIR"]).resolve() in path.parents, (module, path)
    print(f"{module}: {path}")

manager = Path(torch.__file__).resolve().parent / "bin/torch_shm_manager"
assert manager.is_file() and os.access(manager, os.X_OK), manager
assert flash_attn.__version__ == "2.8.3", flash_attn.__version__
q = torch.randn((1, 16, 4, 64), device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, q, q, causal=True)
torch.cuda.synchronize()
assert out.shape == q.shape
print(f"torch={torch.__version__}")
print("SM90 BF16 FlashAttention smoke test: PASS")
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
