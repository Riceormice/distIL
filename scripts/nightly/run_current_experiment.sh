#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

list_jobs() {
  cat <<'EOF'
math_alpha070_rho070
math_alpha070_rho090
math_alpha070_rho095
math_alpha090_rho070
math_alpha090_rho090
math_grpo_4b
math_grpo_8b
math_opsd8x8_4b
math_opsd8x8_8b
physics_logits_sdpo_fkl
physics_logits_sdpo_jsd
EOF
}

if [[ "${1:-}" == "--list" ]]; then
  list_jobs
  exit 0
fi

JOB="${1:?Usage: $0 JOB_NAME; use --list to show names}"
case "${JOB}" in
  math_alpha070_rho070)
    COMMAND=(bash "${REPO}/scripts/math/legacy_all_prompts_h200/h200_alpha070_rho070.sh") ;;
  math_alpha070_rho090)
    COMMAND=(bash "${REPO}/scripts/math/legacy_all_prompts_h200/h200_alpha070_rho090.sh") ;;
  math_alpha070_rho095)
    COMMAND=(bash "${REPO}/scripts/math/legacy_all_prompts_h200/h200_alpha070_rho095.sh") ;;
  math_alpha090_rho070)
    COMMAND=(bash "${REPO}/scripts/math/legacy_all_prompts_h200/h200_alpha090_rho070.sh") ;;
  math_alpha090_rho090)
    COMMAND=(bash "${REPO}/scripts/math/legacy_all_prompts_h200/h200_alpha090_rho090.sh") ;;
  math_grpo_4b)
    COMMAND=(bash "${REPO}/scripts/math/grpo_opsd_trl_aligned/a800_grpo_4b.sh") ;;
  math_grpo_8b)
    COMMAND=(bash "${REPO}/scripts/math/grpo_opsd_trl_aligned/h200_grpo_8b.sh") ;;
  math_opsd8x8_4b)
    COMMAND=(bash "${REPO}/scripts/math/opsd_grouped8x8_a800_4b/a800_opsd_grouped8x8.sh") ;;
  math_opsd8x8_8b)
    COMMAND=(bash "${REPO}/scripts/math/legacy_all_prompts_h200/h200_opsd_grouped8x8.sh") ;;
  physics_logits_sdpo_fkl)
    COMMAND=(bash "${SCRIPT_DIR}/run_p0_sdpo_divergence.sh" sdpo_fkl) ;;
  physics_logits_sdpo_jsd)
    COMMAND=(bash "${SCRIPT_DIR}/run_p0_sdpo_divergence.sh" sdpo_jsd) ;;
  *)
    echo "ERROR: unknown job: ${JOB}" >&2
    list_jobs >&2
    exit 2
    ;;
esac

exec bash "${SCRIPT_DIR}/run_nightly_resumable.sh" "${JOB}" "${COMMAND[@]}"
