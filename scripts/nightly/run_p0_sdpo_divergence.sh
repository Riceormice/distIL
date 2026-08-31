#!/usr/bin/env bash
set -Eeuo pipefail

METHOD="${1:?Usage: $0 sdpo_fkl|sdpo_jsd}"
case "${METHOD}" in
  sdpo_fkl) MACHINE_ID=1 ;;
  sdpo_jsd) MACHINE_ID=2 ;;
  *) echo "ERROR: METHOD must be sdpo_fkl or sdpo_jsd" >&2; exit 2 ;;
esac

P0_REPO="${P0_REPO:-/media/damoxing/che-liu-fileset/ylong/sdpo/code/SDPO-p0-mechanism}"
P0_REQUIRED_COMMIT="${P0_REQUIRED_COMMIT:-535e9814e8c6ce381c72ba0225c29869d93b8e84}"
OUT_ROOT="${P0_OUT_ROOT:-/media/vlm-ckp-fileset/ylong/physics_p0_sdpo_fkl_jsd_20260827}"
MODEL_PATH="${MODEL_PATH:-/media/vlm-ckp-fileset/ylong/sdpo/models/Qwen3-8B}"
METHOD_DIR="${OUT_ROOT}/Qwen3-8B/${METHOD}/seed0"
LAUNCH_CONFIG="${METHOD_DIR}/launch_config.json"
MACHINE_SCRIPT="${P0_REPO}/scripts/physics_mechanism/cluster/sdpo_divergence/machine${MACHINE_ID}_${METHOD}.sh"

if [[ -f "${METHOD_DIR}/state/run.complete" ]]; then
  echo "SKIP: ${METHOD} logits experiment is already complete"
  exit 0
fi

[[ -x "${MACHINE_SCRIPT}" ]] || {
  echo "ERROR: missing P0 launcher: ${MACHINE_SCRIPT}" >&2
  exit 2
}
[[ "$(git -C "${P0_REPO}" rev-parse HEAD 2>/dev/null)" == "${P0_REQUIRED_COMMIT}" ]] || {
  echo "ERROR: P0 repo must be at pinned commit ${P0_REQUIRED_COMMIT}" >&2
  echo "current=$(git -C "${P0_REPO}" rev-parse HEAD 2>/dev/null || echo unavailable)" >&2
  exit 2
}
[[ -z "$(git -C "${P0_REPO}" status --porcelain 2>/dev/null)" ]] || {
  echo "ERROR: P0 repo has local changes; refusing an unreproducible resume" >&2
  git -C "${P0_REPO}" status --short >&2
  exit 2
}

collect_method() {
  export REQUIRE_8_GPUS=0
  source "${P0_REPO}/scripts/physics_mechanism/cluster/runtime_8gpu.sh"
  python "${P0_REPO}/scripts/physics_mechanism/collect_free_generation.py" \
    --method-dir "${METHOD_DIR}" \
    --protocol-dir "${OUT_ROOT}/protocol" \
    --samples-per-question 16 \
    --tokenizer "${MODEL_PATH}"
  [[ -f "${METHOD_DIR}/state/run.complete" ]]
}

if [[ -f "${METHOD_DIR}/state/training.complete" ]]; then
  echo "Training is complete; reconciling capture/generation markers for ${METHOD}"
  collect_method
  echo "COMPLETE: ${METHOD} logits experiment"
  exit 0
fi

if [[ -f "${LAUNCH_CONFIG}" ]]; then
  saved_frequency="$(python3 - "${LAUNCH_CONFIG}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    print(int(json.load(stream)["save_freq"]))
PY
)"
  if [[ -n "${SAVE_FREQ:-}" && "${SAVE_FREQ}" != "${saved_frequency}" ]]; then
    echo "ERROR: existing run uses SAVE_FREQ=${saved_frequency}, got SAVE_FREQ=${SAVE_FREQ}" >&2
    exit 2
  fi
  export SAVE_FREQ="${saved_frequency}"
  echo "Reusing existing checkpoint cadence: SAVE_FREQ=${SAVE_FREQ}"
else
  export SAVE_FREQ="${SAVE_FREQ:-20}"
  echo "New P0 run checkpoint cadence: SAVE_FREQ=${SAVE_FREQ}"
fi

export OUT_ROOT MODEL_PATH
bash "${MACHINE_SCRIPT}"

if [[ -f "${METHOD_DIR}/state/training.complete" ]]; then
  collect_method
  echo "COMPLETE: ${METHOD} logits experiment"
fi
