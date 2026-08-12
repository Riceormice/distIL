# Qwen3-8B mathematics N=16 evaluation on three H200 machines

| Machine | Method | GPUs | Samples per question |
| --- | --- | ---: | ---: |
| 1 | GRPO | 8 H200 | 16 |
| 2 | SDPO | 8 H200 | 16 |
| 3 | SR-OPSD | 8 H200 | 16 |

Each lane evaluates one fixed checkpoint on AIME24, AIME25, HMMT25, AMC23,
and Minerva. All use thinking mode, temperature 1.0, top-p 1.0, top-k -1,
min-p 0, presence penalty 0, 38,912 maximum new tokens, and TP=8.

`CHECKPOINT_DIR` accepts a distIL LoRA checkpoint, a merged Hugging Face model,
a verl `global_step_N` directory, or its `actor` directory. Online loggers are
disabled. The source checkpoint is never changed or removed.
