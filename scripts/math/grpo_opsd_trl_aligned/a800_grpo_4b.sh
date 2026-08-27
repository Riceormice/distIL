#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

exec env \
  REPO="${REPO}" \
  MODEL_SIZE=4b \
  HARDWARE=a800 \
  ENABLE_ADAPTIVE_GPU_KEEPALIVE="${ENABLE_ADAPTIVE_GPU_KEEPALIVE:-1}" \
  OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/math_grpo_4b_opsd_trl_aligned_eval5_n16_a800_20260827}" \
  bash "${SCRIPT_DIR}/run_grpo_opsd_trl_aligned.sh"
