# Nightly resumable experiments

`run_current_experiment.sh` is the single entry point for the experiments that
may run only between 23:00 and 08:00. Each invocation runs for 8 hours 50
minutes, sends `SIGTERM`, waits 10 minutes for cleanup, and then uses `SIGKILL`
only if required. Repeating the same command resumes from the latest complete
checkpoint and skips complete evaluation JSON files.

List supported jobs:

```bash
bash scripts/nightly/run_current_experiment.sh --list
```

Show progress for all current jobs:

```bash
python3 scripts/nightly/show_current_progress.py
```

Run one job:

```bash
bash scripts/nightly/run_current_experiment.sh math_grpo_8b
```

Useful overrides:

```bash
NIGHTLY_WINDOW_SECONDS=31800 \
NIGHTLY_KILL_GRACE_SECONDS=600 \
bash scripts/nightly/run_current_experiment.sh math_grpo_8b
```

The P0 SDPO FKL/JSD wrapper preserves `save_freq` from an existing
`launch_config.json`. A new run defaults to a checkpoint every 20 steps. This
changes checkpoint I/O only; it does not change the training or evaluation
protocol. The wrapper also pins the P0 repository revision; set
`P0_REQUIRED_COMMIT` only when intentionally using a separately validated code
version. Completed runs are skipped before checking this pin. New Physics runs
need a matching fixed probe and the two-step capture/resume acceptance test.

Current native Math runs use lossless shared checkpoint storage. Existing
OPSD/TRL adapter checkpoints keep their existing save format. Training, evaluation,
job IDs and the nightly timeout are unchanged. See
[shared checkpoint migration](../maintenance/SHARED_CHECKPOINTS.md) for both
repository updates, the eight-GPU test, conversion and rollback commands. Local
CPU tests are not a substitute for cluster GPU acceptance. Do not delete the
shared baseline directory or use old checkpoint readers after conversion.
