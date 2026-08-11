#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SDPO_DIR="${REPO_ROOT}/SDPO"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MODEL_SIZE="${MODEL_SIZE:-8b}"
SEED="${SEED:-0}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"
SAVE_FREQ="${SAVE_FREQ:-20}"
SELF_REFERENCE_WEIGHT="${SELF_REFERENCE_WEIGHT:-0.9}"
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/vlm-ckp-fileset/ylong/sr_opsd_verl_math}"
RUN_NAME="${RUN_NAME:-sr-opsd-${MODEL_SIZE}-seed${SEED}-rho0.95-refw${SELF_REFERENCE_WEIGHT}-sync0-lr5e-6-tok16384-steps${TOTAL_STEPS}-native-sdpo-table-aligned}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/checkpoints/${RUN_NAME}}"
MERGED_ROOT="${MERGED_ROOT:-${OUTPUT_ROOT}/merged/${RUN_NAME}}"
RESULT_ROOT="${RESULT_ROOT:-${OUTPUT_ROOT}/evaluations/${RUN_NAME}}"
PHASE="${PHASE:-all}"
KEEP_MERGED_MODELS="${KEEP_MERGED_MODELS:-0}"
KEEP_TRAINING_CHECKPOINTS="${KEEP_TRAINING_CHECKPOINTS:-0}"

export PYTHON_BIN MODEL_SIZE SEED TOTAL_STEPS SAVE_FREQ SELF_REFERENCE_WEIGHT LORA_RANK LORA_ALPHA
export OUTPUT_ROOT RUN_NAME RUN_DIR
export TEST_FREQ="${TEST_FREQ:--1}"
export MAX_ACTOR_CKPT_TO_KEEP="$((TOTAL_STEPS / SAVE_FREQ))"
export PYTHONPATH="${SDPO_DIR}:${PYTHONPATH:-}"

case "${PHASE}" in
  all|train)
    bash "${REPO_ROOT}/scripts/math/run_sr_opsd_verl_math.sh"
    ;;
  eval)
    ;;
  *)
    echo "PHASE must be all, train, or eval" >&2
    exit 2
    ;;
esac

[[ "${DRY_RUN:-0}" == "1" ]] && exit 0
[[ "${PHASE}" == "train" ]] && exit 0

mkdir -p "${MERGED_ROOT}" "${RESULT_ROOT}"
for ((step=SAVE_FREQ; step<=TOTAL_STEPS; step+=SAVE_FREQ)); do
  actor_dir="${RUN_DIR}/global_step_${step}/actor"
  merged_dir="${MERGED_ROOT}/checkpoint-${step}"
  result_dir="${RESULT_ROOT}/checkpoint-${step}"

  complete=1
  for dataset in aime24 aime25 hmmt25 amc23 minerva; do
    if ! [[ -s "${result_dir}/${dataset}.json" ]] || ! \
        "${PYTHON_BIN}" "${REPO_ROOT}/scripts/math/validate_math_eval.py" \
          "${result_dir}/${dataset}.json" --dataset "${dataset}" --samples 64 >/dev/null 2>&1; then
      complete=0
      break
    fi
  done
  if [[ "${complete}" == "1" ]]; then
    echo "SKIP checkpoint-${step}: all five JSON files are complete"
    if [[ "${KEEP_MERGED_MODELS}" == "0" ]]; then
      rm -rf "${merged_dir}"
    fi
    if [[ "${KEEP_TRAINING_CHECKPOINTS}" == "0" ]]; then
      rm -rf "${RUN_DIR}/global_step_${step}"
    fi
    continue
  fi

  test -d "${actor_dir}"

  if [[ ! -s "${merged_dir}/config.json" ]]; then
    rm -rf "${merged_dir}"
    "${PYTHON_BIN}" -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "${actor_dir}" \
      --target_dir "${merged_dir}"
  fi

  lora_adapter_dir=""
  if [[ -s "${merged_dir}/lora_adapter/adapter_config.json" ]]; then
    lora_adapter_dir="${merged_dir}/lora_adapter"
    "${PYTHON_BIN}" - "${lora_adapter_dir}/adapter_config.json" "${LORA_ALPHA}" <<'PY'
import json
import sys

path, alpha = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as handle:
    config = json.load(handle)
config["lora_alpha"] = alpha
with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
  fi

  MODEL_DIR="${merged_dir}" LORA_ADAPTER_DIR="${lora_adapter_dir}" \
    MODEL_SIZE="${MODEL_SIZE}" OUTPUT_DIR="${result_dir}" \
    VAL_N=64 TENSOR_PARALLEL_SIZE=8 \
    bash "${REPO_ROOT}/scripts/math/eval_sr_opsd_verl_math.sh"

  if [[ "${KEEP_MERGED_MODELS}" == "0" ]]; then
    rm -rf "${merged_dir}"
  fi
  if [[ "${KEEP_TRAINING_CHECKPOINTS}" == "0" ]]; then
    rm -rf "${RUN_DIR}/global_step_${step}"
  fi
done

echo "SR-OPSD math pipeline complete: ${RESULT_ROOT}"
