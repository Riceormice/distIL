#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPLOADER="${SCRIPT_DIR}/upload_math_8runs_to_wandb.py"
WATCHER="${SCRIPT_DIR}/watch_math_8runs_wandb.sh"
STATE_ROOT="${STATE_ROOT:-/media/vlm-ckp-fileset/ylong/math_train_eval5_n16_wandb_upload_state}"
PID_FILE="${STATE_ROOT}/periodic_upload.pid"
LOG_FILE="${STATE_ROOT}/periodic_upload.log"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/root/.config/wandb/upload.env}"
ENTITY="${WANDB_ENTITY:-wenxuan-yuan-imperial-college-london}"
PROJECT="${WANDB_PROJECT:-test}"

resolve_python() {
  local candidate
  if [[ -n "${WANDB_PYTHON_BIN:-}" && -x "${WANDB_PYTHON_BIN}" ]]; then
    printf '%s\n' "${WANDB_PYTHON_BIN}"
    return 0
  fi
  for candidate in \
    /media/damoxing/che-liu-fileset/ylong/sdpo/envs/wandb-upload/bin/python \
    /media/damoxing/che-liu-fileset/ylong/sdpo/envs/wandb-uploader/bin/python \
    /media/vlm-ckp-fileset/ylong/sdpo/envs/wandb-upload/bin/python
  do
    if [[ -x "${candidate}" ]] && "${candidate}" -c 'import wandb' >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

load_credentials() {
  if [[ -z "${WANDB_API_KEY:-}" && -r "${WANDB_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${WANDB_ENV_FILE}"
    set +a
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not configured." >&2
    echo "Configure ${WANDB_ENV_FILE} first; the key is never printed by this script." >&2
    exit 2
  fi
  export WANDB_API_KEY
}

is_running() {
  [[ -s "${PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${PID_FILE}")"
  kill -0 "${pid}" 2>/dev/null
}

run_once() {
  local python_bin
  load_credentials
  python_bin="$(resolve_python)" || {
    echo "ERROR: no Python environment containing wandb was found." >&2
    echo "Set WANDB_PYTHON_BIN to the correct interpreter." >&2
    exit 3
  }
  mkdir -p "${STATE_ROOT}/runtime"
  export WANDB_MODE=online
  export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
  export WANDB_DIR="${WANDB_DIR:-${STATE_ROOT}/runtime}"
  export PYTHONUNBUFFERED=1
  exec "${python_bin}" "${UPLOADER}" \
    --entity "${ENTITY}" \
    --project "${PROJECT}" \
    --state-dir "${STATE_ROOT}"
}

case "${1:-status}" in
  start)
    mkdir -p "${STATE_ROOT}"
    load_credentials
    resolve_python >/dev/null || {
      echo "ERROR: no Python environment containing wandb was found." >&2
      exit 3
    }
    if is_running; then
      echo "Watcher already running: pid=$(cat "${PID_FILE}")"
      echo "Dashboard: https://wandb.ai/${ENTITY}/${PROJECT}"
      exit 0
    fi
    rm -f "${PID_FILE}"
    nohup env \
      STATE_ROOT="${STATE_ROOT}" \
      WANDB_ENV_FILE="${WANDB_ENV_FILE}" \
      WANDB_ENTITY="${ENTITY}" \
      WANDB_PROJECT="${PROJECT}" \
      WANDB_PYTHON_BIN="${WANDB_PYTHON_BIN:-}" \
      UPLOAD_INTERVAL_SECONDS="${UPLOAD_INTERVAL_SECONDS:-600}" \
      UPLOAD_TIMEOUT_SECONDS="${UPLOAD_TIMEOUT_SECONDS:-900}" \
      bash "${WATCHER}" </dev/null >/dev/null 2>&1 &
    for _ in $(seq 1 20); do
      if is_running; then
        echo "Watcher started: pid=$(cat "${PID_FILE}")"
        echo "Interval: ${UPLOAD_INTERVAL_SECONDS:-600}s"
        echo "Log: ${LOG_FILE}"
        echo "Dashboard: https://wandb.ai/${ENTITY}/${PROJECT}"
        exit 0
      fi
      sleep 0.5
    done
    echo "ERROR: watcher did not start; inspect ${LOG_FILE}" >&2
    exit 1
    ;;
  stop)
    if ! is_running; then
      rm -f "${PID_FILE}"
      echo "Watcher is not running."
      exit 0
    fi
    pid="$(cat "${PID_FILE}")"
    kill "${pid}"
    for _ in $(seq 1 20); do
      if ! kill -0 "${pid}" 2>/dev/null; then
        echo "Watcher stopped."
        exit 0
      fi
      sleep 0.5
    done
    echo "ERROR: watcher pid ${pid} did not stop cleanly" >&2
    exit 1
    ;;
  restart)
    bash "$0" stop || true
    exec bash "$0" start
    ;;
  status)
    if is_running; then
      echo "Watcher running: pid=$(cat "${PID_FILE}")"
      echo "Dashboard: https://wandb.ai/${ENTITY}/${PROJECT}"
      tail -n 80 "${LOG_FILE}" 2>/dev/null || true
    else
      echo "Watcher is not running."
      tail -n 40 "${LOG_FILE}" 2>/dev/null || true
      exit 1
    fi
    ;;
  once)
    run_once
    ;;
  dry-run)
    python_bin="$(resolve_python)" || python_bin="$(command -v python3)"
    exec "${python_bin}" "${UPLOADER}" \
      --entity "${ENTITY}" \
      --project "${PROJECT}" \
      --state-dir "${STATE_ROOT}" \
      --dry-run
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|once|dry-run}" >&2
    exit 2
    ;;
esac
