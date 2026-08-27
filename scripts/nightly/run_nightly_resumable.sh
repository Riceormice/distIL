#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 JOB_NAME COMMAND [ARG ...]" >&2
  exit 2
fi

JOB_NAME="$1"
shift

[[ "${JOB_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "ERROR: JOB_NAME may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
}

WINDOW_SECONDS="${NIGHTLY_WINDOW_SECONDS:-31800}"
KILL_GRACE_SECONDS="${NIGHTLY_KILL_GRACE_SECONDS:-600}"
STATE_ROOT="${NIGHTLY_STATE_ROOT:-/media/vlm-ckp-fileset/ylong/nightly_experiment_state}"
TIMEOUT_BIN="${TIMEOUT_BIN:-$(command -v timeout || true)}"

[[ "${WINDOW_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: NIGHTLY_WINDOW_SECONDS must be a positive integer" >&2
  exit 2
}
[[ "${KILL_GRACE_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: NIGHTLY_KILL_GRACE_SECONDS must be a positive integer" >&2
  exit 2
}
[[ -x "${TIMEOUT_BIN}" ]] || {
  echo "ERROR: GNU timeout was not found" >&2
  exit 2
}

JOB_ROOT="${STATE_ROOT}/${JOB_NAME}"
mkdir -p "${JOB_ROOT}/logs"
exec 9>"${JOB_ROOT}/nightly.lock"
flock -n 9 || {
  echo "SKIP: ${JOB_NAME} already has a live nightly launcher"
  exit 0
}

LOG="${JOB_ROOT}/logs/$(date +%Y%m%d_%H%M%S).log"
printf '%s\n' "${LOG}" >"${JOB_ROOT}/latest_log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo "nightly_job=${JOB_NAME}"
echo "host=$(hostname)"
echo "started_at=$(date -Iseconds)"
echo "window_seconds=${WINDOW_SECONDS}"
echo "kill_grace_seconds=${KILL_GRACE_SECONDS}"
printf 'command='; printf '%q ' "$@"; printf '\n'
echo "log=${LOG}"
echo "============================================================"

set +e
"${TIMEOUT_BIN}" \
  --signal=TERM \
  --kill-after="${KILL_GRACE_SECONDS}s" \
  "${WINDOW_SECONDS}s" \
  "$@"
status=$?
set -e

{
  echo "status=${status}"
  echo "finished_at=$(date -Iseconds)"
} >"${JOB_ROOT}/last_status.env"

case "${status}" in
  0)
    echo "NIGHTLY RUN COMPLETE: ${JOB_NAME}"
    ;;
  124|137)
    echo "NIGHTLY WINDOW CLOSED: ${JOB_NAME}; rerun the same command next window"
    ;;
  *)
    echo "ERROR: ${JOB_NAME} exited with status ${status}" >&2
    exit "${status}"
    ;;
esac
