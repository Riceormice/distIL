# GRPO recovery and canonical launchers

The previous GRPO failures had two independent causes.

1. The legacy `OPSD/grpo_train.py` path assembled Python, PyTorch, DeepSpeed,
   CUDA, and headers from different directories. DeepSpeed CPUAdam then failed
   to compile because `torch/extension.h` was missing. This was a runtime-stack
   failure, not a GRPO-loss failure.
2. A later native-VERL checkpoint trained successfully, but all inspected
   generations reached the response cap with `finish_reason=length`. Full N=16
   evaluation over five datasets therefore generated an extreme number of
   tokens and appeared stalled. The aligned protocol already bounds GRPO
   evaluation at 16384 tokens; this repair keeps that value unchanged.

`scripts/math/run_grpo_4b.sh` and `scripts/math/run_grpo_8b.sh` now restore the
historically successful OPSD/TRL implementation, but with eight processes and
the current matched protocol. The trainer consumes the local 758-row dataset,
disables training thinking, forces selected checkpoints, stops without
shortening the scheduler horizon, and resumes only a complete checkpoint. It
does not initialize W&B when `report_to=none`.

The pipeline uses the unified OPSD environment, validates checkpoint structure,
and stores an immutable `state/protocol.env`. A resumed run is rejected if any
core training or evaluation field differs. Its output roots are separate from
both the failed native-VERL jobs and older OPSD jobs.

The fixed protocol is: 758 training questions, seed 0, eight GPUs, eight unique
questions by eight rollouts, one policy iteration, GRPO epsilon 0.2, LoRA
64/128, learning rate 5e-6, linear schedule, zero warmup, response length
16384, and training decoding 0.7/0.95/20. It runs 100 physical steps with a
420-step scheduler horizon and evaluates every five steps on five math datasets
with N=16. Qwen3-4B evaluation uses 0.7/0.95/20; Qwen3-8B uses 1.0/1.0/-1.
Both use the 16384-token evaluation cap.
