#!/usr/bin/env bash
set -Eeuo pipefail

SELF_REFERENCE_WEIGHT="${1:?Usage: $0 SELF_REFERENCE_WEIGHT RENYI_ORDER}"
RENYI_ORDER="${2:?Usage: $0 SELF_REFERENCE_WEIGHT RENYI_ORDER}"

case "${SELF_REFERENCE_WEIGHT},${RENYI_ORDER}" in
  0.9,0.7|0.9,0.9|0.7,0.7|0.7,0.9|0.7,0.95) ;;
  *) echo "ERROR: unsupported alpha/rho point: ${SELF_REFERENCE_WEIGHT},${RENYI_ORDER}" >&2; exit 2 ;;
esac

ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-${ROOT}/code/distIL-sr-opsd-renyi}"
SCRIPT_DIR="${REPO}/scripts/math/legacy_all_prompts_h200"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sr_opsd_math_alpha_rho_sweep_legacy_allprompts_20260825}"

source "${SCRIPT_DIR}/common_legacy_all_prompts.sh"
configure_legacy_all_prompts "${OUTPUT_ROOT}"

export ROOT REPO OUTPUT_ROOT
export EVAL_SUBMISSION_MODE EVAL_PROMPT_BATCH_SIZE

exec bash "${REPO}/scripts/math/sweeps/sr_opsd_alpha_rho_8b_h200/run_sr_opsd_8b_alpha_rho_h200.sh" \
  "${SELF_REFERENCE_WEIGHT}" "${RENYI_ORDER}"
