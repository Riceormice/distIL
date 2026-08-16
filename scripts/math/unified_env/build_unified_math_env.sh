#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${1:?Usage: $0 verl|opsd}"
case "${PROFILE}" in
  verl|opsd) ;;
  *) echo "ERROR: profile must be verl or opsd" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-/media/damoxing/che-liu-fileset/conda/bin/conda}"
ENV_HOME="${UNIFIED_ENV_HOME:-/media/damoxing/che-liu-fileset/ylong/sdpo/envs}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/media/vlm-ckp-fileset/ylong/sdpo/cache/pip}"
FLASH_ATTN_SOURCE="${FLASH_ATTN_SOURCE:-/media/vlm-ckp-fileset/ylong/sdpo/build/flash-attn-sm90/src}"
BUILD_ID="${BUILD_ID:-20260817-v1}"
MAX_JOBS="${MAX_JOBS:-4}"

retry() {
  local attempt status
  for attempt in 1 2 3 4 5; do
    set +e
    "$@"
    status=$?
    set -e
    (( status == 0 )) && return 0
    echo "Command failed with status ${status}; retry ${attempt}/5: $*" >&2
    sleep $((attempt * 10))
  done
  return "${status}"
}

case "${PROFILE}" in
  verl)
    TARGET="${ENV_HOME}/math-verl-cu126-py311-${BUILD_ID}"
    ACTIVE="${ENV_HOME}/math-verl-current"
    REQUIREMENTS="${SCRIPT_DIR}/requirements-verl-cu126.txt"
    TORCH_INDEX="https://download.pytorch.org/whl/cu126"
    TORCH_PACKAGES=(torch==2.7.1 torchvision==0.22.1)
    ;;
  opsd)
    TARGET="${ENV_HOME}/math-opsd-cu128-py311-${BUILD_ID}"
    ACTIVE="${ENV_HOME}/math-opsd-current"
    REQUIREMENTS="${SCRIPT_DIR}/requirements-opsd-cu128.txt"
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    TORCH_PACKAGES=(torch==2.8.0 torchvision==0.23.0)
    ;;
esac

mkdir -p "${ENV_HOME}" "${PIP_CACHE_DIR}"
exec 9>"${ENV_HOME}/.build-${PROFILE}.lock"
flock -n 9 || { echo "ERROR: another ${PROFILE} environment build is running" >&2; exit 3; }

LOG="${ENV_HOME}/build-${PROFILE}-$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
trap 'status=$?; echo "ERROR: unified environment build failed: profile='"${PROFILE}"' status=${status} line=${BASH_LINENO[0]} command=${BASH_COMMAND}" >&2; echo "log='"${LOG}"'" >&2; exit "${status}"' ERR

echo "profile=${PROFILE}"
echo "target=${TARGET}"
echo "active=${ACTIVE}"
echo "log=${LOG}"

[[ -x "${CONDA_BIN}" ]] || { echo "ERROR: conda is missing: ${CONDA_BIN}" >&2; exit 2; }
[[ -f "${REQUIREMENTS}" ]] || { echo "ERROR: requirements are missing: ${REQUIREMENTS}" >&2; exit 2; }
if [[ -f "${FLASH_ATTN_SOURCE}/setup.py" || -f "${FLASH_ATTN_SOURCE}/pyproject.toml" ]]; then
  FLASH_ATTN_SPEC="${FLASH_ATTN_SOURCE}"
  echo "FlashAttention source: ${FLASH_ATTN_SPEC}"
else
  FLASH_ATTN_SPEC="flash-attn==2.8.3"
  echo "Local FlashAttention tree is not installable; using ${FLASH_ATTN_SPEC} from the package index"
fi

if [[ -f "${TARGET}/.math-env-complete" && "${REBUILD:-0}" != "1" ]]; then
  echo "Environment already complete; validating before activation"
else
  CREATE_TARGET=1
  if [[ -e "${TARGET}" ]]; then
    if [[ "${RESUME_BUILD:-0}" == "1" ]]; then
      [[ -x "${TARGET}/bin/python" ]] || {
        echo "ERROR: incomplete target has no usable Python; rebuild it with REBUILD=1: ${TARGET}" >&2
        exit 2
      }
      CREATE_TARGET=0
      echo "Resuming incomplete environment: ${TARGET}"
    elif [[ "${REBUILD:-0}" == "1" ]]; then
      rm -rf -- "${TARGET}"
    else
      echo "ERROR: incomplete target exists; rerun with REBUILD=1 after confirming no job uses it: ${TARGET}" >&2
      exit 2
    fi
  fi

  if [[ "${CREATE_TARGET}" == "1" ]]; then
    retry "${CONDA_BIN}" create --yes --copy --prefix "${TARGET}" python=3.11 pip setuptools wheel
  fi
  export PIP_CACHE_DIR PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
  retry "${TARGET}/bin/python" -m pip install --upgrade pip setuptools wheel ninja packaging psutil
  retry "${TARGET}/bin/python" -m pip install --index-url "${TORCH_INDEX}" "${TORCH_PACKAGES[@]}"
  retry "${TARGET}/bin/python" -m pip install --upgrade -r "${REQUIREMENTS}"

  retry env \
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    TORCH_CUDA_ARCH_LIST="8.0;9.0" \
    MAX_JOBS="${MAX_JOBS}" \
    "${TARGET}/bin/python" -m pip install --no-build-isolation --no-deps "${FLASH_ATTN_SPEC}"

  env -u PYTHONHOME -u PYTHONPATH -u CONDA_PREFIX \
    PYTHONNOUSERSITE=1 \
    "${TARGET}/bin/python" -m pip check
  VERIFY_ARGS=(--profile "${PROFILE}" --prefix "${TARGET}" --repo "${REPO}")
  [[ "${GPU_SMOKE:-0}" == "1" ]] && VERIFY_ARGS+=(--gpu-smoke)
  env -u PYTHONHOME -u PYTHONPATH -u CONDA_PREFIX \
    PYTHONNOUSERSITE=1 \
    "${TARGET}/bin/python" "${SCRIPT_DIR}/verify_unified_math_env.py" \
      "${VERIFY_ARGS[@]}"

  "${TARGET}/bin/python" -m pip freeze --all >"${TARGET}/math-env-freeze.txt"
  {
    echo "profile=${PROFILE}"
    echo "build_id=${BUILD_ID}"
    echo "created_at=$(date -Iseconds)"
    echo "repo_commit=$(git -C "${REPO}" rev-parse HEAD)"
    echo "python=$(${TARGET}/bin/python -V 2>&1)"
    echo "torch_index=${TORCH_INDEX}"
    echo "flash_attn_source=${FLASH_ATTN_SPEC}"
    echo "flash_attention_force_build=true"
    echo "cuda_arch_list=8.0;9.0"
  } >"${TARGET}/math-env-manifest.txt"
  touch "${TARGET}/.math-env-complete"
fi

VERIFY_ARGS=(--profile "${PROFILE}" --prefix "${TARGET}" --repo "${REPO}")
[[ "${GPU_SMOKE:-0}" == "1" ]] && VERIFY_ARGS+=(--gpu-smoke)
env -u PYTHONHOME -u PYTHONPATH -u CONDA_PREFIX \
  PYTHONNOUSERSITE=1 \
  "${TARGET}/bin/python" "${SCRIPT_DIR}/verify_unified_math_env.py" \
    "${VERIFY_ARGS[@]}"

LINK_TEMP="${ACTIVE}.tmp.$$"
rm -f -- "${LINK_TEMP}"
ln -s "${TARGET}" "${LINK_TEMP}"
mv -Tf "${LINK_TEMP}" "${ACTIVE}"

echo "Unified ${PROFILE} environment activated: ${ACTIVE} -> ${TARGET}"
echo "Build log: ${LOG}"
