# DistIL: A Distributional DAgger View of Reinforcement Learning from Rich Feedback

This repository contains the code for **DistIL**, a method for reinforcement learning from rich feedback based on a distributional extension of DAgger. DistIL uses a forward cross-entropy objective with sequence-level credit assignment, providing monotonic policy improvement guarantees and improved empirical performance over existing self-distillation methods.

For science and coding experiments, we build on the [SDPO](https://github.com/lasgroup/SDPO) codebase. For mathematical reasoning experiments, we build on the [OPSD](https://github.com/siyan-zhao/OPSD) codebase. We thank the authors of both works for open-sourcing their code.

---

## Repository Structure

```
distil/
├── README.md
├── SDPO/                        # SDPO codebase + DistIL loss (science & coding)
│   ├── verl/trainer/ppo/
│   │   └── core_algos.py        # compute_distil_self_distillation_loss added
│   ├── verl/workers/actor/
│   │   └── dp_actor.py          # loss_mode=distil routing added
│   ├── verl/trainer/config/
│   │   └── distil.yaml          # DistIL config
│   └── ...                      # all other original SDPO files unchanged
├── OPSD/                        # OPSD codebase + DistIL loss (math)
│   ├── opsd_trainer.py          # distil_loss() added, --loss_mode flag
│   ├── opsd_train.py            # --loss_mode: distil | opsd | sdpo
│   └── ...                      # all other original OPSD files unchanged
├── scripts/
│   ├── science/
│   │   ├── run_distil.sh        # DistIL on SciKnowEval 
│   │   ├── run_sdpo.sh          # SDPO baseline
│   │   └── run_grpo.sh          # GRPO baseline
│   ├── coding/
│   │   ├── run_distil.sh        # DistIL on LCBv6
│   │   ├── run_sdpo.sh          # SDPO baseline
│   │   ├── run_grpo.sh          # GRPO baseline
│   │   ├── eval_distil.sh        # DistIL eval on LCBv6
│   │   ├── eval_sdpo.sh          # SDPO eval baseline
│   │   └── eval_grpo.sh          # GRPO eval baseline   
│   └── math/
│       ├── run_distil_4b.sh     # DistIL on Qwen3-4B-Instruct-2507
│       ├── run_distil_8b.sh     # DistIL on Qwen3-8B
│       ├── run_opsd_4b.sh       # OPSD baseline
│       ├── run_opsd_8b.sh
│       ├── run_sdpo_4b.sh       # SDPO baseline (reverse KL)
│       ├── run_sdpo_8b.sh
│       ├── run_sft_4b.sh        # SFT baseline
│       ├── run_sft_8b.sh
│       ├── run_grpo_4b.sh       # GRPO baseline
│       ├── run_grpo_8b.sh
│       └── eval_math.sh         # Evaluate on AIME24/AIME25/HMMT/AMC/Minerva
└── logs/                        # Created automatically on first run
```

---

## Science & Coding Experiments (SDPO-based)

Science and coding experiments use the [SDPO](https://github.com/lasgroup/SDPO) codebase (included in `SDPO/`), extended with the DistIL loss. All scripts are in `scripts/science/` and `scripts/coding/`.

### Installation

```bash
cd SDPO
pip install -e .
```

For full installation instructions (Docker, vLLM, SGLang), see `SDPO/INSTALL.md`.

### Data Preparation

All data preparation commands run from inside `SDPO/`. Detailed instructions are also in `SDPO/data/README.md`.

**Science benchmarks (SciKnowEval L3):**
```bash
cd SDPO

# 1. Download each domain
python data/load_dataset.py --dataset_name Chemistry --output_path datasets/sciknoweval/chemistry.json
python data/load_dataset.py --dataset_name Biology   --output_path datasets/sciknoweval/biology.json
python data/load_dataset.py --dataset_name Material  --output_path datasets/sciknoweval/material.json
python data/load_dataset.py --dataset_name Physics   --output_path datasets/sciknoweval/physics.json

# 2. Split into train/test
python data/split_tasks.py --json_path datasets/sciknoweval/chemistry.json --output_dir datasets/sciknoweval/chemistry --test_ratio 0.1 --seed 42
python data/split_tasks.py --json_path datasets/sciknoweval/biology.json   --output_dir datasets/sciknoweval/biology   --test_ratio 0.1 --seed 42
python data/split_tasks.py --json_path datasets/sciknoweval/material.json  --output_dir datasets/sciknoweval/material  --test_ratio 0.1 --seed 42
python data/split_tasks.py --json_path datasets/sciknoweval/physics.json   --output_dir datasets/sciknoweval/physics   --test_ratio 0.1 --seed 42

# 3. Preprocess to parquet
python data/preprocess.py --data_source datasets/sciknoweval/chemistry
python data/preprocess.py --data_source datasets/sciknoweval/biology
python data/preprocess.py --data_source datasets/sciknoweval/material
python data/preprocess.py --data_source datasets/sciknoweval/physics
```

**Coding benchmark (LCBv6):**
```bash
cd SDPO

# 1. Split unit tests into train/test sets
python data/split_tests.py --json_path datasets/lcb_v6.json --output_dir datasets/lcb_v6

# 2. Preprocess to parquet
python data/preprocess.py --data_source datasets/lcb_v6
```

**Configure paths** — edit `SDPO/verl/trainer/config/user.yaml`:
```yaml
vars:
  dir: /path/to/distil/SDPO       # absolute path to the SDPO directory
  log_dir: /path/to/logs           # directory for logs
  ckpt_dir: /path/to/checkpoints   # directory for model checkpoints
```

### Cluster Configuration

All scripts are fully parameterized. Set the following environment variables before running:

```bash
export ACCOUNT=your_slurm_account
export PARTITION=gpu
export REPO_DIR=/path/to/distil/SDPO
```

Use `--dry-run` to preview commands without submitting:
```bash
bash scripts/science/run_distil.sh --dry-run
```

---

### Science Benchmarks (SciKnowEval L3)

Experiments run on biology, chemistry, materials, and physics with both `Qwen/Qwen3-8B` and `allenai/Olmo-3-7B-Instruct` on 4 × H200 GPUs (~5h per run including validation). Results reported as Avg@16 vs wall-clock time (Table 1 in paper).

| Hyperparameter      | SDPO                      | DistIL (Ours) | GRPO (off/on) |
|---------------------|---------------------------|---|---|
| Learning rate       | 1×10⁻⁵                    | 5×10⁻⁵ | 1×10⁻⁶ / 1×10⁻⁵ |
| Mini-batch size     | 32                        | 32 | 8 / 32 |
| Rollouts            | 8                         | 8 | 8 |
| Top-K distillation  | 100                       | 100 | — |
| Metric              | Jensen–Shannon Divergence | Forward CE | — |
| Teacher update rate | 0.05                      | 0.01 | — |

**DistIL (Ours)** — submits 8 jobs (4 domains × 2 models):
```bash
REPO_DIR=/path/to/distil/SDPO bash scripts/science/run_distil.sh

# Single model only
MODELS="Qwen/Qwen3-8B" REPO_DIR=/path/to/distil/SDPO bash scripts/science/run_distil.sh
```

**SDPO baseline:**
```bash
REPO_DIR=/path/to/distil/SDPO bash scripts/science/run_sdpo.sh
```

**GRPO baseline:**
```bash
# Off-policy (default, as reported in Table 1)
REPO_DIR=/path/to/distil/SDPO bash scripts/science/run_grpo.sh

# On-policy
GRPO_MODE=on_policy REPO_DIR=/path/to/distil/SDPO bash scripts/science/run_grpo.sh
```

---

### Coding Benchmark (LCBv6)

Experiments use `Qwen/Qwen3-8B` with rich execution feedback.

| Hyperparameter | SDPO | DistIL (Ours) | GRPO |
|---|---|---|---|
| Learning rate | 1×10⁻⁶ | 1×10⁻⁶ | 1×10⁻⁶ |
| Mini-batch size | 1 | 1 | 8 |
| Rollouts | 8 | 8 | 8 |
| Top-K distillation | 20 | 20 | — |
| Environment feedback | ✓ | ✓ | ✗ |

**DistIL (Ours):**
```bash
REPO_DIR=/path/to/distil/SDPO bash scripts/coding/run_distil.sh
```

**SDPO baseline:**
```bash
REPO_DIR=/path/to/distil/SDPO bash scripts/coding/run_sdpo.sh
```

**GRPO baseline:**
```bash
REPO_DIR=/path/to/distil/SDPO bash scripts/coding/run_grpo.sh
```

### Evaluation (LCBv6)

Evaluation uses temperature=0.2, top_p=0.95, top_k=20, n=16 as reported in the paper.

```bash
CHECKPOINT=DistIL-coding-lcbv6-Qwen-Qwen3-8B \
CHECKPOINT_STEP=80 \
REPO_DIR=/path/to/distil/SDPO \
bash scripts/coding/eval_distil.sh

CHECKPOINT=SDPO-coding-lcbv6-Qwen-Qwen3-8B  CHECKPOINT_STEP=80 bash scripts/coding/eval_sdpo.sh
CHECKPOINT=GRPO-off_policy-coding-lcbv6-Qwen-Qwen3-8B CHECKPOINT_STEP=80 bash scripts/coding/eval_grpo.sh
```

---

## Math Experiments (OPSD-based)

Mathematical reasoning experiments use the [OPSD](https://github.com/siyan-zhao/OPSD) codebase (included in `OPSD/`), extended with the DistIL loss. All scripts are in `scripts/math/`.

### Installation

```bash
cd OPSD
pip install -e .
```

### Data

The math training set consists of 738 hard problems sourced from [POPE-HARD](https://huggingface.co/datasets/CMU-AIRe/POPE-HARD-w-oracle-solution) and OmniMath (problems with pass@512 = 0 for Qwen3-4B-Instruct), following the setup in our paper. The dataset is provided at:

```
OPSD/data/math/train.jsonl
```

### Training

All training scripts are in `scripts/math/` and can be run from the repo root. Models used Qwen3-4B-Instruct and Qwen3-8B).

| Hyperparameter | GRPO | OPSD / SDPO / DistIL |
|---|---|---|
| Learning rate | 5×10⁻⁶ | 5×10⁻⁶ |
| Effective batch size | 32 | 32 |
| LoRA rank / alpha | 64 / 128 | 64 / 128 |
| Max completion length | 16,000 | 16,384 |
| Training steps | 500 | 100 |
| Top-K distillation | — | 100 |

**DistIL (Ours):**
```bash
bash scripts/math/run_distil_4b.sh   # Qwen3-4B-Instruct-2507
bash scripts/math/run_distil_8b.sh   # Qwen3-8B
```

**OPSD baseline (forward KL / JSD):**
```bash
bash scripts/math/run_opsd_4b.sh
bash scripts/math/run_opsd_8b.sh
```

**SDPO baseline (reverse KL):**
```bash
bash scripts/math/run_sdpo_4b.sh
bash scripts/math/run_sdpo_8b.sh
```

**SFT baseline:**
```bash
bash scripts/math/run_sft_4b.sh
bash scripts/math/run_sft_8b.sh
```

**GRPO baseline:**
```bash
bash scripts/math/run_grpo_4b.sh
bash scripts/math/run_grpo_8b.sh
```

### Overriding Defaults

All scripts accept environment variables:
```bash
MODEL=Qwen/Qwen3-4B-Instruct-2507 \
DATA=OPSD/data/math/train.jsonl \
OUTPUT_DIR=outputs/my_run \
NUM_PROCESSES=4 \
bash scripts/math/run_distil_4b.sh
```

### Evaluation

```bash
# Evaluate base model (no checkpoint)
BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
bash scripts/math/eval_math.sh

# Evaluate a trained checkpoint
BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
CHECKPOINT=outputs/distil_4b/checkpoint-100 \
bash scripts/math/eval_math.sh
```

The eval script applies the correct sampling settings per model (Tables 8 and 9 in the paper):

| Parameter | Qwen3-4B-Instruct-2507 | Qwen3-8B |
|---|---|---|
| Temperature | 0.7 | 1.0 |
| Top-p | 0.95 | 1.0 |
| Top-k | 20 | disabled |
| Max new tokens | 16,384 | 38,912 |
| Thinking mode | enabled | enabled |
| Samples per prompt | 64 | 64 |

Evaluated benchmarks: **AIME24, AIME25, HMMT25, AMC23, Minerva**.

---

## Acknowledgements

We thank [Siyan Zhao](https://github.com/siyan-zhao/OPSD) and co-authors for open-sourcing the OPSD codebase, and the [LAS Group at ETH Zurich](https://github.com/lasgroup/SDPO) for open-sourcing the SDPO codebase. Our implementations of DistIL build directly on top of their work.
