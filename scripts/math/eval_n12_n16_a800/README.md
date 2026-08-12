# Qwen3-8B mathematics evaluation at N=12 and N=16

This queue evaluates two fixed checkpoints on AIME24, AIME25, HMMT25,
AMC23, and Minerva. It does not train or modify either checkpoint.

Use the same exact SDPO checkpoint for machines 1 and 3, and the same exact
SR-OPSD checkpoint for machines 2 and 4:

| Machine | Method | Samples per question |
| --- | --- | ---: |
| 1 | SDPO | 12 |
| 2 | SR-OPSD | 12 |
| 3 | SDPO | 16 |
| 4 | SR-OPSD | 16 |

`CHECKPOINT_DIR` accepts an old distIL LoRA checkpoint, a merged Hugging Face
model, a verl `global_step_N` directory, or its `actor` directory. A verl actor
checkpoint is merged into a temporary model and the temporary merge is removed
after all five result JSON files pass integrity checks.

All four lanes use Qwen3-8B thinking-mode evaluation with temperature 1.0,
top-p 1.0, top-k -1, min-p 0, presence penalty 0, maximum 38,912 new tokens,
and tensor parallelism over all eight A800 GPUs. Online loggers are disabled.
