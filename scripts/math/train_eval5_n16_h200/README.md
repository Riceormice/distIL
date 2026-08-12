# Qwen3-8B mathematics train/evaluate lanes

Four independent 8xH200 lanes train GRPO, SDPO, OPSD, and SR-OPSD through
step 100 while retaining the original 420-step learning-rate horizon. GRPO,
SDPO, and SR-OPSD use the same native SDPO/VERL framework. OPSD keeps its
method-specific distIL/TRL runner. Each lane pauses at steps 5, 10, ..., 100 and evaluates AIME24,
AIME25, HMMT25, AMC23, and Minerva with thinking enabled and 16 samples per
question.

The current checkpoint remains available until the following checkpoint has
been written successfully. The previous checkpoint is then deleted. After the
step-100 JSON files pass completeness validation, the final checkpoint is also
deleted. W&B and SwanLab are disabled; training logs and evaluation JSON files
remain under `/media/vlm-ckp-fileset/ylong/math_train_eval5_n16_h200_20260812`.

The three VERL lanes share Qwen3-8B, seed 0, the same 758-row training set,
prompt/mini-batch sizes 8/8, eight rollouts per question, LoRA rank 64,
response length 16384, temperature/top-p/top-k 0.7/0.95/20, learning rate
5e-6, a linear schedule with no warmup, weight decay 0, gradient clip 0.1,
and token-level rollout correction. Method-specific settings are:

- GRPO: PPO/GRPO clipping 0.2 and group-normalized advantages.
- SDPO: reverse-KL self-distillation, EMA teacher update 0.05, top-k 100,
  no tail bucket, token-loss clip 0.05, and no reference anchoring.
- SR-OPSD: Forward Renyi with rho 0.95, self-reference weight 0.9, frozen
  initial reference (sync 0), plus the same EMA/top-k/clip settings as SDPO.
- OPSD: unchanged dedicated implementation.

Use `launch_h200_machine.sh 1`, `2`, `3`, or `4` for GRPO, SDPO, OPSD, or
SR-OPSD, respectively.
