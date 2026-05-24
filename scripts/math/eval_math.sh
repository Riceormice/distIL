#!/bin/bash
# Resolve OPSD directory relative to this script
OPSD_DIR="$(cd "$(dirname "$0")/../../OPSD" && pwd)"
cd "$OPSD_DIR"
# Evaluate a checkpoint on math benchmarks.
# Uses model-specific eval settings from the paper (Tables 8 and 9).
#
# Usage:
#   # Base model eval (no checkpoint):
#   BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 bash scripts/math/eval_math.sh
#
#   # Checkpoint eval:
#   BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
#   CHECKPOINT=outputs/distil_4b/checkpoint-100 \
#   bash scripts/math/eval_math.sh

BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen3-4B-Instruct-2507"}
CHECKPOINT=${CHECKPOINT:-""}
DATASETS=${DATASETS:-"aime24 aime25 hmmt25 amc23 minerva"}
VAL_N=${VAL_N:-64}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-2}
OUTPUT_DIR=${OUTPUT_DIR:-"eval_results"}

mkdir -p "${OUTPUT_DIR}"

# ── Model-specific eval settings (paper Tables 8 and 9) ──────────────────────
if echo "${BASE_MODEL}" | grep -q "8B"; then
    TEMPERATURE=1.0
    TOP_P=1.0
    TOP_K=-1
    MAX_NEW_TOKENS=38912
    echo "Using Qwen3-8B eval settings: temp=1.0, top_p=1.0, top_k=-1, max_tokens=38912"
else
    # 4B-Instruct-2507
    TEMPERATURE=0.7
    TOP_P=0.95
    TOP_K=20
    MAX_NEW_TOKENS=16384
    echo "Using Qwen3-4B eval settings: temp=0.7, top_p=0.95, top_k=20, max_tokens=16384"
fi
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_ARG=""
[ -n "${CHECKPOINT}" ] && CHECKPOINT_ARG="--checkpoint_dir ${CHECKPOINT}"

for DATASET in ${DATASETS}; do
    echo ""
    echo "========================================================"
    echo "Evaluating ${BASE_MODEL} on ${DATASET}"
    [ -n "${CHECKPOINT}" ] && echo "Checkpoint: ${CHECKPOINT}"
    echo "========================================================"

    CKPT_TAG=""
    [ -n "${CHECKPOINT}" ] && CKPT_TAG="_$(basename ${CHECKPOINT})"
    OUTFILE="${OUTPUT_DIR}/$(basename ${BASE_MODEL})_${DATASET}${CKPT_TAG}.json"

    python eval/evaluate_math.py \
        --base_model "${BASE_MODEL}" \
        ${CHECKPOINT_ARG} \
        --dataset "${DATASET}" \
        --val_n ${VAL_N} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --top_k ${TOP_K} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --tensor_parallel_size ${TENSOR_PARALLEL} \
        --output_file "${OUTFILE}"

    echo "Saved: ${OUTFILE}"
done

echo ""
echo "All done. Results in: ${OUTPUT_DIR}"
