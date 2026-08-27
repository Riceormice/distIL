#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAIN_MANAGER="${MATH_DIR}/wandb_math_8runs/manage_math_8runs_wandb.sh"
SWEEP_UPLOADER="${MATH_DIR}/sweeps/sr_opsd_alpha_rho_8b_h200/upload_alpha_rho_sweep_to_wandb.sh"

ENTITY="${WANDB_ENTITY:-wenxuan-yuan-imperial-college-london}"
PROJECT="${WANDB_PROJECT:-SDPO_math_test}"
STATE_ROOT="${STATE_ROOT:-/media/vlm-ckp-fileset/ylong/sdpo_math_test_current_upload_state}"
MAIN_STATE_ROOT="${MAIN_STATE_ROOT:-${STATE_ROOT}/main_8runs}"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sr_opsd_math_alpha_rho_sweep_20260819}"
SWEEP_STATE_ROOT="${SWEEP_STATE_ROOT:-${SWEEP_OUTPUT_ROOT}/wandb_upload_state_sdpo_math_test}"
LEGACY_SWEEP_OUTPUT_ROOT="${LEGACY_SWEEP_OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sr_opsd_math_alpha_rho_sweep_legacy_allprompts_20260825}"
LEGACY_SWEEP_STATE_ROOT="${LEGACY_SWEEP_STATE_ROOT:-${STATE_ROOT}/alpha_rho_legacy_allprompts}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/root/.config/wandb/upload.env}"
PID_FILE="${STATE_ROOT}/periodic_upload.pid"
LOCK_FILE="${STATE_ROOT}/periodic_upload.lock"
LOG_FILE="${STATE_ROOT}/periodic_upload.log"
INTERVAL_SECONDS="${UPLOAD_INTERVAL_SECONDS:-600}"
TIMEOUT_SECONDS="${UPLOAD_TIMEOUT_SECONDS:-3600}"

mkdir -p \
  "${STATE_ROOT}" \
  "${MAIN_STATE_ROOT}" \
  "${SWEEP_STATE_ROOT}" \
  "${LEGACY_SWEEP_STATE_ROOT}"

run_main() {
  env \
    STATE_ROOT="${MAIN_STATE_ROOT}" \
    WANDB_ENV_FILE="${WANDB_ENV_FILE}" \
    WANDB_ENTITY="${ENTITY}" \
    WANDB_PROJECT="${PROJECT}" \
    bash "${MAIN_MANAGER}" "$@"
}

run_sweep() {
  env \
    OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT}" \
    WANDB_ENV_FILE="${WANDB_ENV_FILE}" \
    WANDB_ENTITY="${ENTITY}" \
    WANDB_PROJECT="${PROJECT}" \
    bash "${SWEEP_UPLOADER}" --state-dir "${SWEEP_STATE_ROOT}" "$@"
}

run_legacy_sweep() {
  env \
    OUTPUT_ROOT="${LEGACY_SWEEP_OUTPUT_ROOT}" \
    WANDB_ENV_FILE="${WANDB_ENV_FILE}" \
    WANDB_ENTITY="${ENTITY}" \
    WANDB_PROJECT="${PROJECT}" \
    bash "${SWEEP_UPLOADER}" \
      --state-dir "${LEGACY_SWEEP_STATE_ROOT}" \
      --variant legacy_allprompts \
      --display-suffix LegacyAllPrompts \
      "$@"
}

is_running() {
  [[ -s "${PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${PID_FILE}")"
  kill -0 "${pid}" 2>/dev/null
}

lock_holders() {
  if command -v fuser >/dev/null 2>&1; then
    fuser "${LOCK_FILE}" 2>/dev/null || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -t "${LOCK_FILE}" 2>/dev/null || true
  fi
}

clear_stale_lock() {
  flock -n "${LOCK_FILE}" -c true 2>/dev/null && return 0

  local holders holder
  holders="$(lock_holders)"
  if [[ -z "${holders//[[:space:]]/}" ]]; then
    echo "ERROR: upload lock is busy but its owner could not be identified: ${LOCK_FILE}" >&2
    return 1
  fi

  echo "Stopping stale upload-lock holders: ${holders}"
  for holder in ${holders}; do
    [[ "${holder}" =~ ^[0-9]+$ ]] && kill -TERM "${holder}" 2>/dev/null || true
  done
  for _ in $(seq 1 30); do
    flock -n "${LOCK_FILE}" -c true 2>/dev/null && return 0
    sleep 1
  done
  echo "ERROR: stale upload lock was not released: ${LOCK_FILE}" >&2
  return 1
}

terminate_watcher() {
  local pid="$1" pgid
  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
  if [[ -n "${pgid}" && "${pgid}" == "${pid}" ]]; then
    kill -TERM -- "-${pgid}" 2>/dev/null || true
  else
    kill -TERM "${pid}" 2>/dev/null || true
  fi
}

show_progress() {
  local failures=0
  echo "===== MAIN + CURRENT GRPO/OPSD RUNS (13 runs) ====="
  run_main dry-run || failures=$((failures + 1))
  echo
  echo "===== 8B SR-OPSD ALPHA/RHO SWEEP (5 runs) ====="
  run_sweep --dry-run || failures=$((failures + 1))
  echo
  echo "===== LEGACY ALL-PROMPTS ALPHA/RHO SWEEP (5 runs) ====="
  run_legacy_sweep --dry-run || failures=$((failures + 1))
  echo
  echo "managed_runs=23 destination=https://wandb.ai/${ENTITY}/${PROJECT} failures=${failures}"
  (( failures == 0 ))
}

upload_once() {
  local failures=0
  echo "===== UPLOAD MAIN + CURRENT GRPO/OPSD RUNS ====="
  run_main once || failures=$((failures + 1))
  echo
  echo "===== UPLOAD ALPHA/RHO 5 RUNS ====="
  run_sweep || failures=$((failures + 1))
  echo
  echo "===== UPLOAD LEGACY ALL-PROMPTS ALPHA/RHO RUNS ====="
  run_legacy_sweep || failures=$((failures + 1))
  echo
  echo "upload_groups=3 managed_runs=23 failures=${failures}"
  (( failures == 0 ))
}

watch_forever() {
  exec 9>"${LOCK_FILE}"
  flock -n 9 || {
    echo "Current Math W&B watcher is already running." >&2
    exit 3
  }
  echo "$$" >"${PID_FILE}"
  trap 'rm -f "${PID_FILE}"' EXIT
  trap 'exit 0' INT TERM

  while true; do
    {
      echo "[$(date -Iseconds)] current Math upload cycle started"
      if timeout --signal=TERM --kill-after=30s "${TIMEOUT_SECONDS}s" \
        env \
          STATE_ROOT="${STATE_ROOT}" \
          MAIN_STATE_ROOT="${MAIN_STATE_ROOT}" \
          SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT}" \
          SWEEP_STATE_ROOT="${SWEEP_STATE_ROOT}" \
          LEGACY_SWEEP_OUTPUT_ROOT="${LEGACY_SWEEP_OUTPUT_ROOT}" \
          LEGACY_SWEEP_STATE_ROOT="${LEGACY_SWEEP_STATE_ROOT}" \
          WANDB_ENV_FILE="${WANDB_ENV_FILE}" \
          WANDB_ENTITY="${ENTITY}" \
          WANDB_PROJECT="${PROJECT}" \
          bash "$0" once 9>&-; then
        echo "[$(date -Iseconds)] current Math upload cycle completed"
      else
        status=$?
        echo "[$(date -Iseconds)] current Math upload cycle failed: exit=${status}"
      fi
    } >>"${LOG_FILE}" 2>&1
    sleep "${INTERVAL_SECONDS}" 9>&- &
    wait $! || true
  done
}

case "${1:-status}" in
  progress|dry-run)
    show_progress
    ;;
  once)
    upload_once
    ;;
  start)
    if is_running; then
      echo "Watcher already running: pid=$(cat "${PID_FILE}")"
      echo "Log: ${LOG_FILE}"
      echo "Dashboard: https://wandb.ai/${ENTITY}/${PROJECT}"
      exit 0
    fi
    rm -f "${PID_FILE}"
    clear_stale_lock || exit 1
    nohup setsid env \
      STATE_ROOT="${STATE_ROOT}" \
      MAIN_STATE_ROOT="${MAIN_STATE_ROOT}" \
      SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT}" \
      SWEEP_STATE_ROOT="${SWEEP_STATE_ROOT}" \
      LEGACY_SWEEP_OUTPUT_ROOT="${LEGACY_SWEEP_OUTPUT_ROOT}" \
      LEGACY_SWEEP_STATE_ROOT="${LEGACY_SWEEP_STATE_ROOT}" \
      WANDB_ENV_FILE="${WANDB_ENV_FILE}" \
      WANDB_ENTITY="${ENTITY}" \
      WANDB_PROJECT="${PROJECT}" \
      UPLOAD_INTERVAL_SECONDS="${INTERVAL_SECONDS}" \
      UPLOAD_TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
      bash "$0" watch </dev/null >>"${LOG_FILE}" 2>&1 &
    for _ in $(seq 1 20); do
      if is_running; then
        echo "Watcher started: pid=$(cat "${PID_FILE}")"
        echo "Interval: ${INTERVAL_SECONDS}s"
        echo "Log: ${LOG_FILE}"
        echo "Dashboard: https://wandb.ai/${ENTITY}/${PROJECT}"
        exit 0
      fi
      sleep 0.5
    done
    echo "ERROR: watcher did not start; inspect ${LOG_FILE}" >&2
    exit 1
    ;;
  watch)
    watch_forever
    ;;
  stop)
    if ! is_running; then
      rm -f "${PID_FILE}"
      echo "Watcher is not running."
      exit 0
    fi
    pid="$(cat "${PID_FILE}")"
    terminate_watcher "${pid}"
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
    else
      echo "Watcher is not running."
    fi
    echo "Dashboard: https://wandb.ai/${ENTITY}/${PROJECT}"
    echo "Log: ${LOG_FILE}"
    tail -n 100 "${LOG_FILE}" 2>/dev/null || true
    ;;
  doctor)
    echo "managed_runs=23"
    echo "main_state=${MAIN_STATE_ROOT}"
    echo "sweep_state=${SWEEP_STATE_ROOT}"
    echo "legacy_sweep_state=${LEGACY_SWEEP_STATE_ROOT}"
    echo "destination=https://wandb.ai/${ENTITY}/${PROJECT}"
    run_main doctor
    ;;
  *)
    echo "Usage: $0 {progress|once|start|stop|restart|status|doctor}" >&2
    exit 2
    ;;
esac
