#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/media/damoxing/che-liu-fileset/ylong/sdpo}"
REPO="${REPO:-${ROOT}/code/distIL-sr-opsd-renyi}"
SCRIPT_DIR="${REPO}/scripts/math/legacy_all_prompts_h200"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/math_opsd_grouped8x8_eval5_n16_h200_legacy_allprompts_20260825}"
RUN_NAME_OVERRIDE="${RUN_NAME_OVERRIDE:-opsd-8b-seed0-grouped-q8-r8-lr5e-6-steps100-sched420-beta0-clip0.06-topk100-temp0.7-tok16384-eval5-n16-h200-legacyallprompts}"

source "${SCRIPT_DIR}/common_legacy_all_prompts.sh"
configure_legacy_all_prompts "${OUTPUT_ROOT}"

export ROOT REPO OUTPUT_ROOT RUN_NAME_OVERRIDE
export EVAL_SUBMISSION_MODE EVAL_PROMPT_BATCH_SIZE

exec bash "${REPO}/scripts/math/opsd_grouped8x8_h200/h200_opsd_grouped8x8.sh"
