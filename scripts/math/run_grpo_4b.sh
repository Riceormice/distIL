#!/usr/bin/env bash
set -Eeuo pipefail

# Canonical Qwen3-4B GRPO entrypoint. The former script used the legacy
# OPSD/TRL stack with four processes and a different optimizer protocol.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

exec env \
  REPO="${REPO}" \
  OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/math_grpo_4b_native_verl_eval5_n16_a800_20260827}" \
  bash "${REPO}/scripts/math/train_eval5_n16_a800_4b/run_verl_method_a800_4b.sh" grpo
