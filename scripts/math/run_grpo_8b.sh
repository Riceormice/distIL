#!/usr/bin/env bash
set -Eeuo pipefail

# Canonical Qwen3-8B GRPO entrypoint. Training and evaluation are delegated to
# the same native VERL pipeline used by the aligned SDPO/SR-OPSD comparison.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

exec env \
  REPO="${REPO}" \
  MODEL_SIZE=8b \
  HARDWARE=h200 \
  OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/math_grpo_8b_native_verl_eval5_n16_h200_20260827}" \
  bash "${REPO}/scripts/math/train_eval5_n16_h200/run_verl_method_h200.sh" grpo
