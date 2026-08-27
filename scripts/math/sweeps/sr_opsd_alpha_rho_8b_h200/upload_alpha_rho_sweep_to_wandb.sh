#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sr_opsd_math_alpha_rho_sweep_20260819}"
ENTITY="${WANDB_ENTITY:-wenxuan-yuan-imperial-college-london}"
PROJECT="${WANDB_PROJECT:-SDPO_math_test}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/root/.config/wandb/upload.env}"

if [[ -z "${WANDB_API_KEY:-}" && -r "${WANDB_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${WANDB_ENV_FILE}"
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not configured." >&2
  echo "Run scripts/wandb/configure_wandb_key.sh interactively first." >&2
  exit 2
fi

if [[ -z "${WANDB_PYTHON_BIN:-}" ]]; then
  for candidate in \
    /media/damoxing/che-liu-fileset/ylong/sdpo/envs/wandb-upload/bin/python \
    /media/vlm-ckp-fileset/ylong/sdpo/envs/wandb-upload/bin/python \
    /media/damoxing/che-liu-fileset/ylong/sdpo/envs/wandb-uploader/bin/python
  do
    if [[ -x "${candidate}" ]] && "${candidate}" -c 'import wandb' >/dev/null 2>&1; then
      WANDB_PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${WANDB_PYTHON_BIN:-}" || ! -x "${WANDB_PYTHON_BIN}" ]]; then
  echo "ERROR: no Python environment containing wandb was found." >&2
  exit 3
fi

"${WANDB_PYTHON_BIN}" - <<'PY'
import os
import re
from importlib.metadata import version

key = os.environ.get("WANDB_API_KEY", "")
match = re.match(r"^(\d+)\.(\d+)", version("wandb"))
installed = tuple(int(value) for value in match.groups()) if match else (0, 0)
if key.startswith("wandb_v1_") and installed < (0, 22):
    raise SystemExit(
        f"wandb {version('wandb')} cannot use wandb_v1 keys; install wandb>=0.22.3"
    )
PY

if [[ ! -d "${OUTPUT_ROOT}/sr_opsd" ]]; then
  echo "WAITING: alpha/rho sweep root is missing: ${OUTPUT_ROOT}/sr_opsd"
fi

export WANDB_MODE=online
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_ENTITY="${ENTITY}"
export WANDB_PROJECT="${PROJECT}"
export PYTHONUNBUFFERED=1

echo "python=${WANDB_PYTHON_BIN}"
echo "source=${OUTPUT_ROOT}"
echo "destination=https://wandb.ai/${ENTITY}/${PROJECT}"
echo "expected_runs=5"

exec "${WANDB_PYTHON_BIN}" "${SCRIPT_DIR}/upload_alpha_rho_sweep_to_wandb.py" \
  --output-root "${OUTPUT_ROOT}" \
  --entity "${ENTITY}" \
  --project "${PROJECT}" \
  "$@"
