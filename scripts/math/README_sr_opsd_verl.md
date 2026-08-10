# SR-OPSD mathematics pipeline

This pipeline trains the true SR-OPSD objective in VERL and evaluates exported
checkpoints with the distIL mathematics evaluator.

## Objective

- Student: optimized policy.
- Self-teacher: exponential moving average of the student, update rate `0.05`.
- Reference: frozen initial policy; synchronization is disabled (`sync=0`).
- Target: normalized geometric mixture with default self-teacher weight `0.9`.
- Projection: forward Renyi divergence with order `rho=0.95`.
- Legacy SDPO selector `alpha` is recorded as `0.25`; the explicit
  `divergence=renyi_forward` setting controls the implemented objective.
- Distribution: student-selected top-100 tokens plus one exact tail-mass bucket.

The actor, EMA teacher, and frozen reference are distinct FSDP models. EMA state
is saved next to every actor checkpoint, so an automatic resume preserves the
training objective.

## Aligned mathematics parameters

- 8 GPUs, seed 0, 100 optimizer steps.
- Checkpoints at steps 20, 40, 60, 80, and 100.
- Effective question batch 8 for 4B and 16 for 8B; eight rollouts per question.
- Maximum prompt/response lengths: 2048/16384.
- Training sampling: temperature 0.7, top-p 0.95, top-k 20.
- AdamW, learning rate `5e-6`, linear decay, zero warmup, zero weight decay,
  gradient norm clip 0.1.
- Local console/JSONL logging only.

Evaluation uses thinking mode, 64 samples per problem, min-p 0, presence
penalty 0, and TP=8. The 4B sampler uses temperature 0.7, top-p 0.95, top-k
20, and 16384 new tokens; the 8B sampler uses temperature 1.0, top-p 1.0,
disabled top-k (`-1`), and 38912 new tokens. The five datasets are AIME24,
AIME25, HMMT25, AMC23, and Minerva.

## Run

```bash
PYTHON_BIN=/path/to/verl/bin/python \
MODEL_SIZE=8b \
MODEL_PATH=/path/to/Qwen3-8B \
NUM_GPUS=8 \
bash scripts/math/run_sr_opsd_verl_math_pipeline.sh
```

Set `PHASE=train` or `PHASE=eval` to run one phase. Both phases are restartable;
training uses VERL auto-resume and evaluation skips only JSON files that pass
the problem/sample-count validator.

`SELF_REFERENCE_WEIGHT` overrides the self-teacher weight without changing any
other method or optimization parameter. For example, set it to `0.95`, `0.9`,
`0.85`, or `0.8` for the four-lane coefficient sweep.
