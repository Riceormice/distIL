#!/usr/bin/env bash

# Source this file as: source activate_unified_math_env.sh verl|opsd
PROFILE="${1:?Usage: source activate_unified_math_env.sh verl|opsd}"
case "${PROFILE}" in
  verl|opsd) ;;
  *) echo "ERROR: profile must be verl or opsd" >&2; return 2 ;;
esac

UNIFIED_ENV_HOME="${UNIFIED_ENV_HOME:-/media/damoxing/che-liu-fileset/ylong/sdpo/envs}"
case "${PROFILE}" in
  verl) DEFAULT_ENV_DIR="${UNIFIED_ENV_HOME}/math-verl-current" ;;
  opsd) DEFAULT_ENV_DIR="${UNIFIED_ENV_HOME}/math-opsd-current" ;;
esac
ENV_DIR="${ENV_DIR:-${DEFAULT_ENV_DIR}}"

[[ -f "${ENV_DIR}/.math-env-complete" ]] || {
  echo "ERROR: unified ${PROFILE} environment is not complete: ${ENV_DIR}" >&2
  return 2
}
[[ -x "${ENV_DIR}/bin/python" ]] || {
  echo "ERROR: unified ${PROFILE} Python is missing: ${ENV_DIR}/bin/python" >&2
  return 2
}

unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV
export ENV_DIR
export PYTHON_BIN="${ENV_DIR}/bin/python"
export PATH="${ENV_DIR}/bin:/usr/local/cuda/bin:/usr/bin:/bin:${PATH:-}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${ENV_DIR}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

unset PROFILE DEFAULT_ENV_DIR
