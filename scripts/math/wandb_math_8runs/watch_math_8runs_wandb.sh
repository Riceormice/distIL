#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_ROOT="${STATE_ROOT:-/media/vlm-ckp-fileset/ylong/math_train_eval5_n16_wandb_upload_state}"
LOG_FILE="${STATE_ROOT}/periodic_upload.log"
LOCK_FILE="${STATE_ROOT}/periodic_upload.lock"
PID_FILE="${STATE_ROOT}/periodic_upload.pid"
INTERVAL_SECONDS="${UPLOAD_INTERVAL_SECONDS:-600}"
UPLOAD_TIMEOUT_SECONDS="${UPLOAD_TIMEOUT_SECONDS:-900}"
MAX_LOG_BYTES="${MAX_UPLOAD_LOG_BYTES:-20971520}"

mkdir -p "${STATE_ROOT}" "${STATE_ROOT}/runtime"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "The Math 8-run W&B watcher is already running." >&2
  exit 3
fi

echo "$$" >"${PID_FILE}"
sleep_pid=""
cleanup() {
  if [[ -n "${sleep_pid}" ]]; then
    kill "${sleep_pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
}
trap cleanup EXIT
trap 'exit 0' INT TERM

rotate_log_if_needed() {
  local size=0
  if [[ -f "${LOG_FILE}" ]]; then
    size="$(stat -c '%s' "${LOG_FILE}" 2>/dev/null || echo 0)"
  fi
  if (( size >= MAX_LOG_BYTES )); then
    mv -f "${LOG_FILE}" "${LOG_FILE}.1"
  fi
}

run_upload() {
  rotate_log_if_needed
  {
    echo "[$(date -Iseconds)] upload cycle started"
    if timeout --signal=TERM --kill-after=30s \
      "${UPLOAD_TIMEOUT_SECONDS}s" \
      env WANDB_DIR="${STATE_ROOT}/runtime" \
      WANDB_SILENT=true \
      WANDB_CONSOLE=off \
      STATE_ROOT="${STATE_ROOT}" \
      bash "${SCRIPT_DIR}/manage_math_8runs_wandb.sh" once; then
      echo "[$(date -Iseconds)] upload cycle completed"
    else
      status=$?
      echo "[$(date -Iseconds)] upload cycle failed: exit=${status}"
    fi
  } >>"${LOG_FILE}" 2>&1
}

while true; do
  run_upload
  sleep "${INTERVAL_SECONDS}" 9>&- &
  sleep_pid=$!
  wait "${sleep_pid}" || true
  sleep_pid=""
done
