# Periodic W&B upload for the current eight Math runs

This directory uploads the four Qwen3-8B H200 runs and four Qwen3-4B A800
runs to `wenxuan-yuan-imperial-college-london/test`. It reads the shared
filesystem only and never modifies training outputs.

The uploader handles GRPO, SDPO, OPSD, and SR-OPSD under these roots:

- `/media/vlm-ckp-fileset/ylong/math_train_eval5_n16_h200_20260812`
- `/media/vlm-ckp-fileset/ylong/math_4b_train_eval5_n16_a800_20260812`

Training metrics are uploaded incrementally. A dataset evaluation is uploaded
only after its JSON is complete and passes strict validation for the expected
problem count and 16 samples per problem. Training and evaluation use separate
W&B step metrics, so evaluations that finish later are not dropped.

Start the 10-minute watcher on the development machine:

```bash
UPLOAD_INTERVAL_SECONDS=600 bash \
  /media/damoxing/che-liu-fileset/ylong/sdpo/code/distIL-sr-opsd-renyi/scripts/math/wandb_math_8runs/manage_math_8runs_wandb.sh start
```

Inspect it with `status`, diagnose credentials/runtime selection with `doctor`,
run one immediate cycle with `once`, and terminate it with `stop`. The W&B key
is loaded from `/root/.config/wandb/upload.env` and is never printed. New
`wandb_v1_...` secrets are automatically routed to W&B 0.22 or newer.
