#!/usr/bin/env bash
set -Eeuo pipefail

# Canonical Qwen3-8B GRPO entrypoint. It uses the historical OPSD/TRL stack
# with the aligned eight-question by eight-rollout Math protocol.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

exec env REPO="${REPO}" \
  bash "${REPO}/scripts/math/grpo_opsd_trl_aligned/h200_grpo_8b.sh"
