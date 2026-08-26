# Qwen3-4B OPSD grouped 8x8 on A800

This lane uses eight A800 GPUs and the dedicated OPSD/TRL implementation. Each
optimizer step samples eight distinct math questions and eight independent
rollouts for each question, giving 64 trajectories. The launcher verifies the
758-row dataset hash and the distributed sampler layout before training.

The comparison protocol is unchanged: seed 0, learning rate 5e-6, linear
schedule with zero warmup, 100 physical steps with a 420-step scheduler horizon,
response length 16384, temperature/top-p/top-k 0.7/0.95/20, LoRA 64/128, and
external evaluation every five steps. Evaluation uses AIME24, AIME25, HMMT25,
AMC23, and Minerva with thinking enabled, N=16, temperature/top-p/top-k
0.7/0.95/20, and max_new_tokens 16384.

Run:

```bash
exec bash /media/damoxing/che-liu-fileset/ylong/sdpo/code/distIL-sr-opsd-renyi/scripts/math/opsd_grouped8x8_a800_4b/a800_opsd_grouped8x8.sh
```
