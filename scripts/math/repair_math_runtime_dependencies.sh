#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-${ROOT}/code/distIL-sr-opsd-renyi}"
VERL_ENV="${VERL_ENV:-/media/vlm-ckp-fileset/ylong/sdpo/envs/verl-vllm010-h200-v2}"
OPSD_ENV="${OPSD_ENV:-/media/vlm-ckp-fileset/ylong/sdpo/envs/opsd-math}"
CONDA_ROOT="${CONDA_ROOT:-/media/damoxing/che-liu-fileset/conda}"
OVERLAY="${MATH_DEPENDENCY_REPAIR_OVERLAY:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_overlays/math_dependency_repair_20260816}"

CORE_RUNTIME="${CORE_RUNTIME:-/media/vlm-ckp-fileset/ylong/sdpo/runtime/math-core-torch2.8-ray2.50.1-v1}"
PYTHON_EXTRAS="${PYTHON_EXTRAS:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_assets/math-python-extras-v1}"
PYTHON_COMPLETE="${PYTHON_COMPLETE:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_assets/math-python-complete-v2}"
VLLM_COMPLETE="${VLLM_COMPLETE:-/media/vlm-ckp-fileset/ylong/sdpo/runtime_assets/math-vllm-python-complete-v1}"
OPSD_SITE="${OPSD_ENV}/lib/python3.11/site-packages"

for path in "${VERL_ENV}/bin/python" "${OPSD_ENV}/bin/python"; do
  [[ -x "${path}" ]] || { echo "ERROR: missing Python interpreter: ${path}" >&2; exit 2; }
done

declare -a SEARCH_ROOTS=()
add_root() {
  local root="$1" existing
  [[ -d "${root}" ]] || return 0
  for existing in "${SEARCH_ROOTS[@]:-}"; do
    [[ "${existing}" == "${root}" ]] && return 0
  done
  SEARCH_ROOTS+=("${root}")
}

# Prefer known complete environments, then inspect every shared Python 3.11 env.
add_root /media/vlm-ckp-fileset/ylong/sdpo/envs/verl-vllm010-official/lib/python3.11/site-packages
add_root /media/damoxing/che-liu-fileset/ylong/sdpo/envs/opsd-math/lib/python3.11/site-packages
add_root /media/vlm-ckp-fileset/ylong/sdpo/envs/opsd-math/lib/python3.11/site-packages
add_root "${CONDA_ROOT}/lib/python3.11/site-packages"
for root in \
  /media/vlm-ckp-fileset/ylong/sdpo/envs/*/lib/python3.11/site-packages \
  /media/damoxing/che-liu-fileset/ylong/sdpo/envs/*/lib/python3.11/site-packages
do
  add_root "${root}"
done

select_root() {
  local package="$1" sentinel="$2" require_binary="${3:-false}" root
  for root in "${SEARCH_ROOTS[@]}"; do
    [[ -f "${root}/${package}/${sentinel}" ]] || continue
    if [[ "${require_binary}" == true ]] &&
       ! compgen -G "${root}/${package}/_raylet*.so" >/dev/null; then
      continue
    fi
    printf '%s\n' "${root}"
    return 0
  done
  echo "ERROR: no complete ${package} package found (missing ${sentinel})" >&2
  return 1
}

RAY_ROOT="$(select_root ray _common/utils.py true)"
DEEPSPEED_ROOT="$(select_root deepspeed runtime/engine.py)"
ACCELERATE_ROOT="$(select_root accelerate utils/dataclasses.py)"

TEMP="${OVERLAY}.tmp.$$"
rm -rf -- "${TEMP}"
mkdir -p "${TEMP}"
ln -s "${RAY_ROOT}/ray" "${TEMP}/ray"
ln -s "${DEEPSPEED_ROOT}/deepspeed" "${TEMP}/deepspeed"
ln -s "${ACCELERATE_ROOT}/accelerate" "${TEMP}/accelerate"

# Keep package metadata consistent with the package code selected above.
for package_spec in \
  "${RAY_ROOT}/ray-*.dist-info" \
  "${DEEPSPEED_ROOT}/deepspeed-*.dist-info" \
  "${ACCELERATE_ROOT}/accelerate-*.dist-info"
do
  for metadata_dir in ${package_spec}; do
    [[ -d "${metadata_dir}" ]] || continue
    ln -s "${metadata_dir}" "${TEMP}/$(basename "${metadata_dir}")"
  done
done

echo "ray_source=${RAY_ROOT}" >"${TEMP}/sources.env"
echo "deepspeed_source=${DEEPSPEED_ROOT}" >>"${TEMP}/sources.env"
echo "accelerate_source=${ACCELERATE_ROOT}" >>"${TEMP}/sources.env"

echo "===== VERL Ray validation ====="
env -u PYTHONHOME -u CONDA_PREFIX \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${REPO}/SDPO:${REPO}:${TEMP}" \
  "${VERL_ENV}/bin/python" - <<'PY'
from pathlib import Path
import ray
import ray._common.utils

print(f"ray={ray.__version__} {Path(ray.__file__).resolve()}")
print(f"ray._common.utils={Path(ray._common.utils.__file__).resolve()}")
PY

echo "===== OPSD Accelerate/DeepSpeed validation ====="
OPSD_PYTHONPATH="${TEMP}:${ROOT}/code/distIL/OPSD:${ROOT}/code/distIL:${PYTHON_EXTRAS}:${VLLM_COMPLETE}:${PYTHON_COMPLETE}:${CORE_RUNTIME}:${OPSD_SITE}"
PYTHONHOME="${CONDA_ROOT}" PYTHONNOUSERSITE=1 PYTHONPATH="${OPSD_PYTHONPATH}" \
  "${OPSD_ENV}/bin/python" - <<'PY'
from pathlib import Path
import accelerate.utils.dataclasses
import deepspeed.runtime.engine

print(f"accelerate.utils.dataclasses={Path(accelerate.utils.dataclasses.__file__).resolve()}")
print(f"deepspeed.runtime.engine={Path(deepspeed.runtime.engine.__file__).resolve()}")
PY

touch "${TEMP}/.complete"
rm -rf -- "${OVERLAY}.previous"
if [[ -e "${OVERLAY}" ]]; then
  mv "${OVERLAY}" "${OVERLAY}.previous"
fi
mv "${TEMP}" "${OVERLAY}"
rm -rf -- "${OVERLAY}.previous"

echo "Dependency repair overlay ready: ${OVERLAY}"
cat "${OVERLAY}/sources.env"
