# Native SDPO SR-OPSD mathematics training

This branch contains the mathematics training path from the SDPO `math-train`
fork. Training uses VERL; the distIL code is retained only for the independent
five-benchmark evaluator.

## Objective

- `alpha=0.25` selects Forward Renyi in the native SDPO implementation.
- `rho=0.95` is the Renyi order.
- `renyi_regularization=True` enables the frozen-reference target.
- `renyi_regularization_level=0.9` is the default self-reference weight.
- `renyi_ref_sync_steps=0` keeps the reference at the initial policy.
- The self-teacher is an EMA model with update rate `0.01`.
- The student, EMA teacher, and frozen reference are distinct FSDP models.

The EMA teacher is saved beside every actor checkpoint so that automatic resume
preserves the training state.

## Data

The author-provided JSONL files are committed under
`SDPO/datasets/math_probs`. Generate the VERL parquet files with:

```bash
PYTHON_BIN=/path/to/verl/bin/python bash SDPO/prepare_math_data.sh
```

Both JSONL files contain 758 records. They are used for in-training Math reward
and validation only; final reporting remains the external AIME24, AIME25,
HMMT25, AMC23, and Minerva evaluation.

## Direct training

```bash
PYTHON_BIN=/path/to/verl/bin/python \
MODEL_PATH=/path/to/Qwen3-8B \
NUM_GPUS=8 \
TOTAL_STEPS=100 \
TEST_FREQ=5 \
SAVE_FREQ=20 \
TRAINER_LOGGER='[console,file]' \
bash run_local_ours_math.sh
```

Defaults follow the supplied script: LoRA rank 64/alpha 128, batch size 32,
eight rollouts, learning rate `5e-6`, constant schedule with 10 warmup steps,
prompt/response lengths 2048/8192, rollout temperature 0.8, top-p 0.95,
and 15 maximum epochs (the explicit step limit remains authoritative). Every
value can be overridden through an environment variable.

## Training plus five-benchmark evaluation

The compatibility wrapper uses full-parameter training (`LORA_RANK=0`) because
the external evaluator expects a merged Hugging Face model:

```bash
PYTHON_BIN=/path/to/verl/bin/python \
MODEL_SIZE=8b \
MODEL_PATH=/path/to/Qwen3-8B \
NUM_GPUS=8 \
bash scripts/math/run_sr_opsd_verl_math_pipeline.sh
```

Set `PHASE=train` or `PHASE=eval` to run only one phase. Evaluation uses
thinking mode and 64 samples per problem across all five Math benchmarks.
