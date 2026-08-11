#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/math_probs}"

test -x "${PYTHON_BIN}"
test -f "${DATA_DIR}/train.json"
test -f "${DATA_DIR}/test.json"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" data/preprocess.py --data_source "${DATA_DIR}"

test -s "${DATA_DIR}/train.parquet"
test -s "${DATA_DIR}/test.parquet"

"${PYTHON_BIN}" - "${DATA_DIR}" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

data_dir = Path(sys.argv[1])
for split in ("train", "test"):
    path = data_dir / f"{split}.parquet"
    table = pq.read_table(path, columns=["data_source", "prompt", "reward_model"])
    if table.num_rows == 0:
        raise SystemExit(f"empty parquet: {path}")
    required = {"data_source", "prompt", "reward_model"}
    if not required.issubset(table.column_names):
        raise SystemExit(f"invalid parquet schema: {path}: {table.column_names}")
    print(f"{split}: rows={table.num_rows}, path={path}")
PY
