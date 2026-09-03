#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_UPLOADER="${SCRIPT_DIR}/../sr_opsd_alpha_rho_8b_h200/upload_alpha_rho_sweep_to_wandb.sh"

OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sr_opsd_math_4b_alpha_rho_sweep_eval5_n16_a800_20260829}"
STATE_DIR="${WANDB_STATE_DIR:-/media/vlm-ckp-fileset/ylong/sdpo_math_test_current_upload_state/alpha_rho_4b_a800}"
ENTITY="${WANDB_ENTITY:-wenxuan-yuan-imperial-college-london}"
PROJECT="${WANDB_PROJECT:-SDPO_math_test}"

if [[ ! -x "${SHARED_UPLOADER}" ]]; then
  echo "ERROR: shared W&B uploader is missing: ${SHARED_UPLOADER}" >&2
  exit 2
fi

echo "Qwen3-4B Math SR-OPSD alpha/rho upload"
echo "source=${OUTPUT_ROOT}"
echo "state=${STATE_DIR}"
echo "destination=https://wandb.ai/${ENTITY}/${PROJECT}"
echo "grid=(0.9,0.7) (0.9,0.9) (0.7,0.7) (0.7,0.9) (0.7,0.95)"

exec env \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  WANDB_ENTITY="${ENTITY}" \
  WANDB_PROJECT="${PROJECT}" \
  bash "${SHARED_UPLOADER}" \
    --state-dir "${STATE_DIR}" \
    --model-size 4B \
    --hardware A800 \
    --variant legacy_allprompts \
    --display-suffix LegacyAllPrompts \
    "$@"
