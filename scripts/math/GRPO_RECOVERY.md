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

`scripts/math/run_grpo_4b.sh` and `scripts/math/run_grpo_8b.sh` are now canonical
wrappers around the native VERL train/evaluate pipeline. They no longer invoke
the obsolete four-process TRL launcher. The pipeline uses a self-contained
runtime preflight, validates checkpoint structure and tokenizer/EOS metadata,
and stores an immutable `state/protocol.env`. A resumed run is rejected if any
core training or evaluation field differs. Their default output roots are new
dated directories, so suspect legacy checkpoint-5 files are not silently reused.

The fixed protocol is: 758 training questions, seed 0, eight GPUs, train batch
8, mini-batch 8, eight rollouts per question, learning rate 5e-6, linear
schedule, zero warmup, response length 16384, and training decoding
0.7/0.95/20. It runs 100 physical steps with a 420-step scheduler horizon and
evaluates every five steps on five math datasets with N=16. Qwen3-4B evaluation
uses 0.7/0.95/20 and 16384 tokens; Qwen3-8B GRPO evaluation uses 1.0/1.0/-1 and
16384 tokens.
