# Aligned OPSD/TRL GRPO

These launchers restore the previously successful `OPSD/grpo_train.py` stack
without reusing the invalid native-VERL GRPO outputs.

Both runs use the same 758 Math questions, eight unique prompts with eight
rollouts per optimizer step, LoRA 64/128, LR 5e-6, a 420-step linear scheduler,
and stop after 100 physical steps. Five Math datasets are evaluated at N=16
after every five optimizer steps. Evaluated checkpoints are deleted after the
next resumable checkpoint is available; the final checkpoint is deleted after
its evaluation completes. Numeric Trainer logs are retained in
`training_metrics.jsonl` even after checkpoint cleanup.

LoRA dropout is fixed to 0 to match the native VERL adapters. The historical
OPSD/TRL GRPO loss has no entropy-bonus term, so `entropy_coefficient=0` is
recorded explicitly in each run protocol. This is the only intentional loss
configuration difference from the native VERL Math launchers, which used the
very small coefficient `1e-5`.

The Qwen3-4B run targets eight A800 GPUs and enables an adaptive utilization
keepalive because that cluster previously evicted low-utilization jobs. The
Qwen3-8B run targets eight H200/H20Z GPUs. The keepalive changes neither model
updates nor sampled trajectories.

```bash
exec bash scripts/math/grpo_opsd_trl_aligned/a800_grpo_4b.sh
exec bash scripts/math/grpo_opsd_trl_aligned/h200_grpo_8b.sh
```

Rerunning the same command resumes the highest complete checkpoint and skips
validated evaluation files. Set `PRECHECK_ONLY=1` to run environment, GPU,
model, dataset, and evaluator checks without training.
