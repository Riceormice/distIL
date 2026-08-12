# Qwen3-8B mathematics train/evaluate lanes

Four independent 8xH200 lanes train GRPO, SDPO, OPSD, and SR-OPSD through
step 100 while retaining the original 420-step learning-rate horizon. Each
lane pauses at steps 5, 10, ..., 100 and evaluates AIME24,
AIME25, HMMT25, AMC23, and Minerva with thinking enabled and 16 samples per
question.

The current checkpoint remains available until the following checkpoint has
been written successfully. The previous checkpoint is then deleted. After the
step-100 JSON files pass completeness validation, the final checkpoint is also
deleted. W&B and SwanLab are disabled; training logs and evaluation JSON files
remain under `/media/vlm-ckp-fileset/ylong/math_train_eval5_n16_h200_20260812`.
