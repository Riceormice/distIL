#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
exec env REPO="${REPO}" \
  bash "${REPO}/scripts/math/grpo_opsd_trl_aligned/a800_grpo_4b.sh"
