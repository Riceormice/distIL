# Qwen3-8B mathematics train/evaluate lanes

Four independent 8xH200 lanes train GRPO, SDPO, OPSD, and SR-OPSD through
step 100 while retaining the original 420-step learning-rate horizon. SDPO and
SR-OPSD use native SDPO/VERL. GRPO and OPSD use their method-specific
OPSD/TRL implementations. Each lane pauses at steps 5, 10, ..., 100 and evaluates AIME24,
AIME25, HMMT25, AMC23, and Minerva with thinking enabled and 16 samples per
question.

The current checkpoint remains available until the following checkpoint has
been written successfully. The previous checkpoint is then deleted. After the
step-100 JSON files pass completeness validation, the final checkpoint is also
deleted. W&B and SwanLab are disabled; training logs and evaluation JSON files
remain under method-specific output roots. The restored GRPO root is
`/media/vlm-ckp-fileset/ylong/math_grpo_8b_opsd_trl_aligned_eval5_n16_h200_20260827`.

The methods share Qwen3-8B, seed 0, the same 758-row training set,
eight questions and eight rollouts per optimizer step, LoRA rank 64,
response length 16384, temperature/top-p/top-k 0.7/0.95/20, learning rate
5e-6, a linear schedule with no warmup, weight decay 0, gradient clip 0.1,
and token-level rollout correction. Method-specific settings are:

- GRPO: OPSD/TRL GRPO with clipping 0.2, one policy iteration, global token-mean
  aggregation, and group-normalized rewards.
- SDPO: reverse-KL self-distillation, EMA teacher update 0.05, top-k 100,
  no tail bucket, token-loss clip 0.05, and no reference anchoring.
- SR-OPSD: Forward Renyi with rho 0.95, self-reference weight 0.9, frozen
  initial reference (sync 0), plus the same EMA/top-k/clip settings as SDPO.
- OPSD: dedicated grouped implementation with eight questions and eight
  completions per question.

Use `launch_h200_machine.sh 1`, `2`, `3`, or `4` for GRPO, SDPO, OPSD, or
SR-OPSD, respectively.
