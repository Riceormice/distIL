#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sdpo_physics_rho_selfref_grid_eval5_nockpt}"
ENTITY="${WANDB_ENTITY:-wenxuan-yuan-imperial-college-london}"
PROJECT="${WANDB_PROJECT:-sdpo_ablation_physics}"

find_python_with_wandb() {
  local candidate
  for candidate in \
    "${PYTHON_BIN:-}" \
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

if [[ ! -d "${OUTPUT_ROOT}/logs" ]]; then
  echo "ERROR: Physics result directory is missing: ${OUTPUT_ROOT}/logs" >&2
  exit 2
fi

if ! PYTHON_BIN="$(find_python_with_wandb)"; then
  echo "ERROR: no Python environment containing wandb was found." >&2
  echo "Install wandb in an available environment, then set PYTHON_BIN explicitly." >&2
  exit 3
fi

export WANDB_MODE=online
export WANDB_ENTITY="${ENTITY}"
export WANDB_PROJECT="${PROJECT}"
export PYTHONUNBUFFERED=1

echo "python=${PYTHON_BIN}"
echo "source=${OUTPUT_ROOT}"
echo "destination=https://wandb.ai/${ENTITY}/${PROJECT}"
echo "expected_runs=8"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/upload_physics_grid_to_wandb.py" \
  --output-root "${OUTPUT_ROOT}" \
  --entity "${ENTITY}" \
  --project "${PROJECT}" \
  "$@"
