# Qwen3-4B mathematics on four 8xA800 machines

The four lanes run GRPO, SDPO, OPSD, and SR-OPSD, respectively. All lanes
train to physical step 100 using a 420-step linear learning-rate horizon and
pause every five steps for external evaluation. Each evaluation uses thinking
mode and 16 samples per problem on AIME24, AIME25, HMMT25, AMC23, and Minerva.

Shared training settings are Qwen3-4B, seed 0, learning rate
5e-6, no warmup, weight decay 0, gradient clipping 0.1, response length 16384,
temperature/top-p/top-k 0.7/0.95/20, and LoRA rank 64. The three native VERL
lanes use eight rollouts and train/mini-batch sizes 8/8. OPSD uses its dedicated
distIL runner with one completion per training example, per-device batch 1,
and gradient accumulation 1.

Method-specific objectives are GRPO with epsilon 0.2; SDPO reverse KL with an
EMA teacher; OPSD with beta 0; and SR-OPSD Forward Renyi with rho 0.95,
self-reference weight 0.9, and a frozen initial reference. Online W&B and
SwanLab logging are disabled.

Evaluated checkpoints and temporary merged models are deleted. Validated JSON
results and local training logs remain under
`/media/vlm-ckp-fileset/ylong/math_4b_train_eval5_n16_a800_20260812`.
