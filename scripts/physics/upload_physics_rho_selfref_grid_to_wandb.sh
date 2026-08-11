#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sdpo_physics_rho_selfref_grid_eval5_nockpt}"
ENTITY="${WANDB_ENTITY:-wenxuan-yuan-imperial-college-london}"
PROJECT="${WANDB_PROJECT:-sdpo_ablation_physics}"
WANDB_UPLOAD_ENV="${WANDB_UPLOAD_ENV:-/media/damoxing/che-liu-fileset/ylong/sdpo/envs/wandb-upload}"
WANDB_VERSION="${WANDB_VERSION:-0.17.9}"
WANDB_PIP_TIMEOUT="${WANDB_PIP_TIMEOUT:-300}"
WANDB_PIP_RETRIES="${WANDB_PIP_RETRIES:-10}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/root/.config/wandb/upload.env}"

if [[ -z "${WANDB_API_KEY:-}" && -r "${WANDB_ENV_FILE}" ]]; then
  # The file is created with mode 600 by configure_wandb_key.sh.
  source "${WANDB_ENV_FILE}"
fi

find_python_with_wandb() {
  local candidate
  for candidate in \
    "${PYTHON_BIN:-}" \
    "${WANDB_UPLOAD_ENV}/bin/python" \
    /media/vlm-ckp-fileset/ylong/sdpo/envs/verl-vllm010-h200-v2/bin/python \
    /media/vlm-ckp-fileset/ylong/sdpo/envs/verl-vllm010-official/bin/python \
    /media/vlm-ckp-fileset/ylong/sdpo/envs/opsd-math/bin/python \
    "$(command -v python3 2>/dev/null || true)"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    if "${candidate}" -c 'import wandb' >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

bootstrap_wandb_environment() {
  local base_python=""
  local candidate
  local index_url
  local -a index_urls

  for candidate in \
    /media/damoxing/che-liu-fileset/conda/bin/python \
    /media/damoxing/che-liu-fileset/conda/bin/python3 \
    /usr/bin/python3 \
    "$(command -v python3 2>/dev/null || true)"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    if "${candidate}" -c 'import venv' >/dev/null 2>&1; then
      base_python="${candidate}"
      break
    fi
  done

  if [[ -z "${base_python}" ]]; then
    echo "ERROR: no Python interpreter with the venv module was found." >&2
    return 1
  fi

  echo "No existing Python environment contains wandb." >&2
  echo "Creating an isolated upload environment at ${WANDB_UPLOAD_ENV}" >&2
  mkdir -p "$(dirname "${WANDB_UPLOAD_ENV}")"
  if [[ ! -x "${WANDB_UPLOAD_ENV}/bin/python" ]]; then
    "${base_python}" -m venv "${WANDB_UPLOAD_ENV}" >&2
  fi

  if [[ -n "${WANDB_PIP_INDEX_URL:-}" ]]; then
    index_urls=("${WANDB_PIP_INDEX_URL}")
  else
    index_urls=(
      https://pypi.tuna.tsinghua.edu.cn/simple
      https://mirrors.aliyun.com/pypi/simple
      https://pypi.org/simple
    )
  fi

  for index_url in "${index_urls[@]}"; do
    echo "Installing wandb==${WANDB_VERSION} from ${index_url}" >&2
    if "${WANDB_UPLOAD_ENV}/bin/python" -m pip install \
      --disable-pip-version-check \
      --prefer-binary \
      --retries "${WANDB_PIP_RETRIES}" \
      --timeout "${WANDB_PIP_TIMEOUT}" \
      --index-url "${index_url}" \
      "wandb==${WANDB_VERSION}" >&2; then
      break
    fi
    echo "WARNING: installation from ${index_url} failed; trying the next index." >&2
  done

  if ! "${WANDB_UPLOAD_ENV}/bin/python" -c 'import wandb' >/dev/null 2>&1; then
    echo "ERROR: wandb installation failed on every configured package index." >&2
    return 1
  fi
  printf '%s\n' "${WANDB_UPLOAD_ENV}/bin/python"
}

if [[ ! -d "${OUTPUT_ROOT}/logs" ]]; then
  echo "ERROR: Physics result directory is missing: ${OUTPUT_ROOT}/logs" >&2
  exit 2
fi

if ! PYTHON_BIN="$(find_python_with_wandb)"; then
  if ! PYTHON_BIN="$(bootstrap_wandb_environment)"; then
    echo "ERROR: unable to prepare the isolated W&B upload environment." >&2
    echo "Set WANDB_PIP_INDEX_URL to a reachable Python package index and retry." >&2
    exit 3
  fi
fi

export WANDB_MODE=online
export WANDB_ENTITY="${ENTITY}"
export WANDB_PROJECT="${PROJECT}"
export PYTHONUNBUFFERED=1

echo "python=${PYTHON_BIN}"
echo "source=${OUTPUT_ROOT}"
echo "destination=https://wandb.ai/${ENTITY}/${PROJECT}"
echo "expected_runs=8"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/upload_physics_rho_selfref_grid_to_wandb.py" \
  --output-root "${OUTPUT_ROOT}" \
  --entity "${ENTITY}" \
  --project "${PROJECT}" \
  "$@"
