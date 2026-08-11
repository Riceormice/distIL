# SR-OPSD Mathematics Protocol Comparison

Both H200 lanes train Qwen3-8B for 30 optimizer steps on the same `math_probs`
data. In-process validation on `math_probs/test.parquet` runs every five steps
with one validation rollout and does not create checkpoints. Only step 30 is
saved temporarily, then evaluated with thinking enabled, 64 samples per
problem, and TP=8 on AIME24, AIME25, HMMT25, AMC23, and Minerva. After all five
JSON files pass validation, both the training checkpoint and merged model are
deleted. An incomplete checkpoint is retained only when external evaluation
fails, so the failed evaluation can be resumed without retraining.

`table_aligned` uses the Mathematics SR-OPSD settings reported in the current
configuration table: one rollout per question, 16384 training response tokens,
linear learning-rate decay without warmup, full-model training, no tail bucket,
no distillation IS clip, token-loss clip 0.05, and teacher student-update
fraction 0.05.

`github_original` preserves the defaults in the original `math-train` branch
`run_local_ours_math.sh`: eight rollouts, 8192 response tokens, constant
learning rate with 10 warmup steps, LoRA rank 64, tail bucket, distillation IS
clip 2.0, no token-loss clip, and teacher student-update fraction 0.01.

The extra checkpoints and five-dataset evaluation are observational only and
do not change either optimization protocol.
