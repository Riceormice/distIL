#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-${ROOT}/code/distIL-sr-opsd-renyi}"
VERL_ENV="${VERL_ENV:-/media/vlm-ckp-fileset/ylong/sdpo/envs/verl-vllm010-h200-v2}"
PYTHON_BIN="${PYTHON_BIN:-/media/damoxing/che-liu-fileset/conda/bin/python}"
DEPENDENCY_REPAIR_OVERLAY="${MATH_DEPENDENCY_REPAIR_OVERLAY:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_overlays/math_dependency_repair_20260816}"
OVERLAY="${MATH_NATIVE_VERL_HF_OVERLAY:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_overlays/math_native_verl_hf_20260817}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/media/vlm-ckp-fileset/ylong/sdpo/cache/pip}"

export SETUPTOOLS_USE_DISTUTILS=stdlib
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -x "${VERL_ENV}/bin/python" ]] || { echo "ERROR: missing VERL Python: ${VERL_ENV}/bin/python" >&2; exit 2; }
[[ -f "${DEPENDENCY_REPAIR_OVERLAY}/.complete" ]] || {
  echo "ERROR: base dependency repair is incomplete: ${DEPENDENCY_REPAIR_OVERLAY}" >&2
  exit 2
}

TEMP="${OVERLAY}.tmp.$$"
rm -rf -- "${TEMP}"
mkdir -p "${TEMP}" "${PIP_CACHE_DIR}"

echo "Installing isolated native VERL Hugging Face stack"
PYTHONPATH="${DEPENDENCY_REPAIR_OVERLAY}" \
  PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
  "${PYTHON_BIN}" -m pip install \
    --no-deps \
    --upgrade \
    --target "${TEMP}" \
    'transformers==4.57.1' \
    'tokenizers==0.22.2'

echo "===== Native VERL HF validation ====="
env -u PYTHONHOME -u CONDA_PREFIX \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${TEMP}:${DEPENDENCY_REPAIR_OVERLAY}:${REPO}/SDPO:${REPO}" \
  "${VERL_ENV}/bin/python" - <<'PY'
from importlib.metadata import version
from pathlib import Path

import tokenizers
import transformers

assert transformers.__version__ == "4.57.1", transformers.__version__
assert tokenizers.__version__ == "0.22.2", tokenizers.__version__
assert version("transformers") == "4.57.1", version("transformers")
assert version("tokenizers") == "0.22.2", version("tokenizers")
assert any(Path(tokenizers.__file__).parent.glob("tokenizers*.so"))
print(f"transformers={transformers.__version__} {Path(transformers.__file__).resolve()}")
print(f"tokenizers={tokenizers.__version__} {Path(tokenizers.__file__).resolve()}")
PY

printf '%s\n' \
  'transformers=4.57.1' \
  'tokenizers=0.22.2' \
  >"${TEMP}/versions.env"
touch "${TEMP}/.complete"

rm -rf -- "${OVERLAY}.previous"
if [[ -e "${OVERLAY}" ]]; then
  mv "${OVERLAY}" "${OVERLAY}.previous"
fi
mv "${TEMP}" "${OVERLAY}"
rm -rf -- "${OVERLAY}.previous"

echo "Native VERL HF overlay ready: ${OVERLAY}"
cat "${OVERLAY}/versions.env"
