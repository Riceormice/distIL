# SR-OPSD Mathematics Protocol Comparison

Both H200 lanes train Qwen3-8B for 30 optimizer steps on the same `math_probs`
data. Model-only snapshots are written at steps 5, 10, 15, 20, 25, and 30;
optimizer state, extra state, and the EMA teacher are not saved. After training
releases all eight GPUs, every snapshot is evaluated by the independent distIL
evaluator on AIME24, AIME25, HMMT25, AMC23, and Minerva with thinking enabled,
64 samples per problem, and TP=8. Once all five JSON files pass validation, the
corresponding snapshot and merged model are deleted. An incomplete snapshot is
retained only when its external evaluation fails.

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
