#!/usr/bin/env bash
set -Eeuo pipefail

WANDB_ENV_FILE="${WANDB_ENV_FILE:-/root/.config/wandb/upload.env}"

if [[ "${1:-}" == "--check" ]]; then
  if [[ -r "${WANDB_ENV_FILE}" ]] && grep -q '^export WANDB_API_KEY=' "${WANDB_ENV_FILE}"; then
    echo "W&B key is configured: ${WANDB_ENV_FILE}"
    exit 0
  fi
  echo "W&B key is not configured: ${WANDB_ENV_FILE}" >&2
  exit 1
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  key="${WANDB_API_KEY}"
elif [[ -t 0 ]]; then
  read -rsp "W&B API key: " key
  echo
else
  echo "ERROR: no interactive terminal and WANDB_API_KEY is unset." >&2
  echo "Configure WANDB_API_KEY as a platform environment variable, or run this script in WebIDE." >&2
  exit 2
fi

key="${key//$'\r'/}"
key="${key//$'\n'/}"
if [[ ${#key} -eq 37 ]]; then
  echo "ERROR: the supplied 37-character value is a W&B Key ID, not the full secret." >&2
  echo "Create a new API key and copy the one-time secret from the creation dialog." >&2
  exit 3
fi
if [[ ${#key} -lt 40 || "${key}" == *[[:space:]]* ]]; then
  echo "ERROR: the supplied W&B API key is incomplete or malformed." >&2
  exit 3
fi

umask 077
mkdir -p "$(dirname "${WANDB_ENV_FILE}")"
printf 'export WANDB_API_KEY=%q\n' "${key}" > "${WANDB_ENV_FILE}"
chmod 600 "${WANDB_ENV_FILE}"
unset key

echo "W&B key configured securely: ${WANDB_ENV_FILE}"
echo "The Physics W&B uploader will load this file automatically."
