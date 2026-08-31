# OPSD/SDPO Checkpoint Retention Audit

Run `audit_opsd_checkpoint_retention.py` on the development server to identify
weight directories separately from experiment evidence. This is read-only and
has no `--apply` option. Only reports are written, outside the data root.

## Scope

- Current Math roots come from `scripts/nightly/show_current_progress.py`: the
  five 8B and five 4B alpha/rho runs, OPSD/TRL GRPO 4B/8B, and grouped OPSD 4B/8B.
- Historical Math roots are explicitly allowlisted in the audit. Their absence
  from the current default launchers is not proof that no custom job uses them.
- The three known Physics analysis/ablation roots are included, with raw logits
  and all analysis evidence protected.
- MARQ, mcRL, DWRQ, shared `sdpo` assets, base models, environments, source code,
  result archives and other projects are not scanned.

The audit finds step directories inside `checkpoints`/`merged`, and entire
`model_states` trees. It does not follow symlinks or nested mounts, inspect
unrecognized weight layouts as deletion candidates, or load tensors/pickles.

## Retention

| Label | Meaning |
| --- | --- |
| `KEEP_CURRENT_LATEST` | Preserve the latest current checkpoint even when the nightly job is paused. |
| `KEEP_NEWER_UNVERIFIED` | A newer checkpoint lacks basic resume files; preserve older fallback checkpoints. |
| `KEEP_BUSY` | A run or nightly lock is held; do not clean its weights. |
| `KEEP_EVALUATION_PENDING` | The merged export may still be needed for evaluation. |
| `KEEP_UNMAPPED` / `KEEP_LOCK_UNKNOWN` | Missing ownership/activity evidence; retain. |
| `REVIEW_CURRENT_OLDER` | Possible old checkpoint; verify newer restore AND old-step evaluation before removal. |
| `REVIEW_HISTORICAL_WEIGHTS` | Old default experiment; confirm retirement and no remaining resume dependencies. |
| `REVIEW_EVALUATED_EXPORT` | Five result files exist; validate their contents and evaluator inactivity. |
| `REVIEW_COMPLETED_RUN` | Completion marker exists; validate all required evaluations before final removal. |
| `REVIEW_PHYSICS_WEIGHTS` | Use the separately validated Physics cleanup tool. |

`REVIEW` does not authorize deletion. Resume file hints are not a real restore
test. A newly interrupted checkpoint can have the highest step but be incomplete;
retain the previous usable checkpoint until the newer one is verified. Keep all
optimizer/RNG/data-loader state inside the retained checkpoint, not just weights.

Evaluation file counts are existence checks, not correctness/completeness
validation. Never delete a checkpoint needed to finish a missing evaluation.
No local PID or log-age heuristic is used to declare a remote job stopped.
Lock observations are point-in-time evidence, not a guarantee about future runs.

Keep evaluations, logs, configs, completion/upload state, generations,
`raw_logits_audit`, `topk_probe`, token stats, aggregates, and figures.
The existing current launchers already remove older checkpoints after the next
checkpoint and required evaluation are ready; audit leftovers before changing
that policy.

## Usage

```bash
REPO=/media/damoxing/che-liu-fileset/ylong/sdpo/code/distIL-sr-opsd-renyi
REPORT="$HOME/workspace/opsd_storage_audit_$(date +%Y%m%d_%H%M%S)"
python3 -u "$REPO/scripts/maintenance/audit_opsd_checkpoint_retention.py" \
  --report-dir "$REPORT"
```

`summary.txt` lists concrete checkpoint paths and the largest retained data.
`checkpoints.tsv`, `retained_data.tsv`, and `audit.json` contain the full report.
The report directory must be new and outside `/media/vlm-ckp-fileset/ylong`.
Review any `scan_errors`; unknown sizes or depth limits prevent a complete audit.
Top-level sizes include evidence and are not potential savings. Hardlinks and
shared storage mean even candidate allocated-byte totals are not guaranteed
reclaimed space.

For the five completed Physics mechanism runs, use
`cleanup_verified_physics_weights.py` after its evidence checks pass; do not
replace its checks with a completion-marker-only `rm -rf`.
