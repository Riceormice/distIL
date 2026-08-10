#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to a merged Hugging Face checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for the five evaluation JSON files}"
MODEL_SIZE="${MODEL_SIZE:-8b}"
VAL_N="${VAL_N:-64}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
DATASETS=(aime24 aime25 hmmt25 amc23 minerva)

case "${MODEL_SIZE}" in
  4b)
    TEMPERATURE=0.7
    TOP_P=0.95
    TOP_K=20
    MAX_NEW_TOKENS=16384
    ;;
  8b)
    TEMPERATURE=1.0
    TOP_P=1.0
    TOP_K=-1
    MAX_NEW_TOKENS=38912
    ;;
  *)
    echo "MODEL_SIZE must be 4b or 8b, got ${MODEL_SIZE}" >&2
    exit 2
    ;;
esac

test -f "${MODEL_DIR}/config.json"
mkdir -p "${OUTPUT_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

missing=()
for dataset in "${DATASETS[@]}"; do
  result="${OUTPUT_DIR}/${dataset}.json"
  if [[ -s "${result}" ]] && "${PYTHON_BIN}" "${REPO_ROOT}/scripts/math/validate_math_eval.py" \
      "${result}" --dataset "${dataset}" --samples "${VAL_N}" >/dev/null 2>&1; then
    echo "SKIP complete ${result}"
  else
    rm -f "${result}"
    missing+=("${dataset}")
  fi
done

if (( ${#missing[@]} > 0 )); then
  "${PYTHON_BIN}" "${REPO_ROOT}/OPSD/eval/evaluate_math.py" \
    --base_model "${MODEL_DIR}" \
    --datasets "${missing[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --enable_thinking \
    --val_n "${VAL_N}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --min_p 0 \
    --presence_penalty 0 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_model_len 40960 \
    --gpu_memory_utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.90}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
fi

for dataset in "${DATASETS[@]}"; do
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/math/validate_math_eval.py" \
    "${OUTPUT_DIR}/${dataset}.json" --dataset "${dataset}" --samples "${VAL_N}"
done
