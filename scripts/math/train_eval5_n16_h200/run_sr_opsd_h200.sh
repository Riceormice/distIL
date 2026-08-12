#!/usr/bin/env bash
set -Eeuo pipefail

# Backward-compatible entrypoint. New runs share the native VERL pipeline used
# by GRPO and SDPO.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_verl_method_h200.sh" sr_opsd
