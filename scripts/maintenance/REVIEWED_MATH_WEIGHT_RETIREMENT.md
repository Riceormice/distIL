# Reviewed Math Weight Retirement (2026-08-31)

Source: the user-supplied `summary.txt` from
`/root/workspace/opsd_storage_audit_20260831_215035` (47 weight entries, no scan errors).
This script retires old runs at the user's request to keep only current experiments.
It does **not** assert that all historical runs finished or have complete evaluations.
Deleting these weights prevents continuing their old trajectories or re-evaluating them.
Old and new runs with the same alpha/rho are not necessarily the same training trajectory.

## Scope

| Historical weights selected | Directories | Report size (GiB) |
|---|---:|---:|
| Old Aug19 sweep: (.7,.7), (.9,.9), (.7,.95) only | 5 | 224.16 |
| Aug11 original-protocol comparison | 7 | 206.64 |
| Aug12 8B obsolete SR-OPSD, native GRPO, OPSD ga2 | 5 | 188.72 |
| Aug12 4B native GRPO | 2 | 26.91 |
| Aug27 failed 8B native GRPO | 2 | 48.74 |
| Aug27 failed 4B native GRPO | 2 | 26.91 |
| Aug20 chunked OPSD grouped 8x8 | 1 | 2.62 |
| Total | 24 | 724.70 |

These are sums of rounded allocated-size measurements, not guaranteed reclaimed
space. Hardlinks, shared storage and changes since the audit can reduce savings.

Kept unconditionally:

- Old alpha=.9/rho=.7 step60 versus current step50; old alpha=.7/rho=.9 step65
  versus current step55. Four weight directories, about 159.84 GiB, await a decision.
- Every current Math run: 8B legacy-all-prompts sweep, 4B Aug29 sweep, new
  OPSD/TRL GRPO4B/8B, grouped OPSD4B Aug27 and grouped OPSD8B legacy-all-prompts.
- All Physics weights/data, raw logits/top-K arrays, metrics, evaluations,
  generations, logs, outside-checkpoint configs and upload state; all other projects.
- Current weights and pending evaluation exports total about 584.55 GiB in the
  report. Absence of new GRPO checkpoints from that list is not a job-status claim.

## Safeguards

The code contains 24 exact relative paths, not a glob-based deletion rule and not
an automatic interpretation of `REVIEW_HISTORICAL_WEIGHTS`. Unexpected new step
directories in a reviewed weight family block deletion of that run's candidates.
The nightly job registry is checked again to reject any root reclassified current.

The three retired sweep pairs additionally require their current checkpoints at
or above the reviewed current step, with nonempty data and all eight actor model,
optimizer and extra-state shards. This checks structure, not successful tensor restore.

The tool takes shared-filesystem cleanup locks and existing run pipeline locks;
it rejects recently modified weights/logs/evaluations, symlinks, nested mounts,
special files and evidence subdirectories inside weight trees. Inventories are
checked again before deletion. Per-path receipts preserve file metadata and small
checkpoint JSON/YAML/TXT files before fd-based `rmtree` is called.

Locks do not establish global inactivity. **Stop any custom job using these old
paths before applying, and do not relaunch old commands during cleanup.** Some
original-protocol scripts never took a pipeline lock. Current nightly jobs need
not be abandoned; none of their weight directories are selected.

## Run On The Development Machine

No GPU, Torch or W&B dependency is needed. Python 3.10+ on Linux is sufficient.
Keep receipts on the development machine's local disk, outside `ylong`.

```bash
REPO=/media/damoxing/che-liu-fileset/ylong/sdpo/code/distIL-sr-opsd-renyi
REPORT="$HOME/workspace/math_retirement_dry_$(date +%Y%m%d_%H%M%S)"
python3 -u "$REPO/scripts/maintenance/cleanup_reviewed_math_weights.py" \
  --report-dir "$REPORT"
```

After agreeing to retire the listed old trajectories (not the held two sweeps):

```bash
REPORT="$HOME/workspace/math_retirement_apply_$(date +%Y%m%d_%H%M%S)"
python3 -u "$REPO/scripts/maintenance/cleanup_reviewed_math_weights.py" \
  --report-dir "$REPORT" --apply --retire-listed-historical-runs
```

`DELETED` means that exact directory was removed; `ABSENT` is harmless on rerun.
`FAILED_OR_SKIPPED` means inspect the stated condition; do not bypass it with a
broad deletion command. If deletion was interrupted, inspect `events.jsonl` and
receipts before retrying. The script does not rewrite old checkpoint pointers,
resume configuration or training code. Do not reuse retired runs as current jobs.
