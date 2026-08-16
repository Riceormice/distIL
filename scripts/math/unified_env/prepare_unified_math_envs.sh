#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building the unified native VERL environment"
GPU_SMOKE="${GPU_SMOKE:-1}" bash "${SCRIPT_DIR}/build_unified_math_env.sh" verl

echo "Building the unified OPSD/distIL environment"
GPU_SMOKE="${GPU_SMOKE:-1}" bash "${SCRIPT_DIR}/build_unified_math_env.sh" opsd

echo "============================================================"
echo "Unified math environments are ready"
readlink -f /media/damoxing/che-liu-fileset/ylong/sdpo/envs/math-verl-current
readlink -f /media/damoxing/che-liu-fileset/ylong/sdpo/envs/math-opsd-current
echo "============================================================"
