#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sr_opsd_math_refw_sweep30}"
ENTITY="${WANDB_ENTITY:-wenxuan-yuan-imperial-college-london}"
PROJECT="${WANDB_PROJECT:-SDPO_table2}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/root/.config/wandb/upload.env}"
PYTHON_BIN="${WANDB_PYTHON_BIN:-/media/damoxing/che-liu-fileset/ylong/sdpo/envs/wandb-upload/bin/python}"

if [[ -z "${WANDB_API_KEY:-}" && -r "${WANDB_ENV_FILE}" ]]; then
  source "${WANDB_ENV_FILE}"
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not configured." >&2
  echo "Run scripts/wandb/configure_wandb_key.sh from an interactive terminal." >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: W&B upload Python is missing: ${PYTHON_BIN}" >&2
  exit 3
fi
if ! "${PYTHON_BIN}" -c '
from importlib.metadata import version
parts = tuple(int(value) for value in version("wandb").split(".")[:3])
raise SystemExit(0 if parts >= (0, 22, 3) else 1)
' >/dev/null 2>&1; then
  echo "ERROR: this uploader requires wandb>=0.22.3 in ${PYTHON_BIN}." >&2
  exit 4
fi
if [[ ! -d "${OUTPUT_ROOT}/evaluations" ]]; then
  echo "ERROR: H200 Math result root is incomplete: ${OUTPUT_ROOT}" >&2
  exit 5
fi

export WANDB_MODE=online
export WANDB_ENTITY="${ENTITY}"
export WANDB_PROJECT="${PROJECT}"
export PYTHONUNBUFFERED=1

echo "python=${PYTHON_BIN}"
echo "source=${OUTPUT_ROOT}"
echo "destination=https://wandb.ai/${ENTITY}/${PROJECT}"
echo "expected_runs=4"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/upload_sr_opsd_refw30_h200_to_wandb.py" \
  --output-root "${OUTPUT_ROOT}" \
  --entity "${ENTITY}" \
  --project "${PROJECT}" \
  "$@"
