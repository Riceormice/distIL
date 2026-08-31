# Completed Physics Weight Cleanup

`cleanup_verified_physics_weights.py` is an explicit five-run allowlist, not a
general cleanup command. It covers the completed seed-0 Physics mechanism runs:

- `physics_p0_mechanism_20260825_v2`: `opsd_fkl`, `sdpo_rkl`, `sr_opsd`.
- `physics_p0_sdpo_fkl_jsd_20260827`: `sdpo_fkl`, `sdpo_jsd`.

Only each run's `Qwen3-8B/METHOD/seed0/model_states` directory can be deleted.
The user authorized removal of these completed checkpoints on 2026-08-31.
Their reported directory usage was approximately 640.90 GiB in total; shared
files/hardlinks may make actual reclaimed space smaller.

## Run on the Development Machine

Use the existing Python 3; no training environment or GPU is required.
Do not force-relaunch these completed Physics jobs during cleanup. Normal P0
launchers reject completed runs, and nightly launchers skip them. Existing
pipeline and FKL/JSD nightly locks are respected; missing locks are not proof
that every remote machine is idle.

```bash
python3 scripts/maintenance/cleanup_verified_physics_weights.py \
  --report-dir "$HOME/workspace/physics_cleanup_preview_$(date +%Y%m%d_%H%M%S)"

python3 scripts/maintenance/cleanup_verified_physics_weights.py \
  --report-dir "$HOME/workspace/physics_cleanup_apply_$(date +%Y%m%d_%H%M%S)" \
  --apply
```

Report directories must be new and outside the experimental data root. Before
deleting, the script checks 420-step/eval5 completion metadata, matching probe
IDs, all 85 capture/generation markers, saved generation/evaluation/token-stat
files, all rank-level top-K and audit shard paths, and the aggregate files.
It checks presence and nonempty sizes, not the numeric validity of NPZ/Parquet
contents. Preserve the prior collector validation results.

It rejects symlinks, nested mounts, recent weight writes, and protected analysis
directory names inside the weight tree. A second inventory check detects changes
during validation. Receipts record the weight inventory and retained evidence
metadata before deletion; the JSONL journal records start and completion. It
does not make a backup of the weights. Once deleted, those checkpoints cannot
be used to resume training or collect new model outputs.

Raw logits, generations, aggregate tables, metrics, configs and completion
markers are retained. Math runs, older uncertain checkpoints, MARQ, DWRQ,
mcRL, runtime environments, base models, and result archives are not selected.
`FAILED_OR_SKIPPED` requires inspecting the reason; do not bypass the checks.
After an interruption, rerun with a new report directory. Absent weight trees
are skipped; surviving partial trees must pass the same checks again.
