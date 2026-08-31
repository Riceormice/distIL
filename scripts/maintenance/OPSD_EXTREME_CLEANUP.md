# Expanded Math Storage Cleanup

This is project-only cleanup, not deletion of everything under `ylong`. It does
not change training, evaluation, checkpoint serialization or loading code.

## Deletes

- With `--retire-historical`, weights in the eight historical roots listed in
  `audit_opsd_checkpoint_retention.HISTORICAL_ROOTS`. This abandons those old
  trajectories, including unfinished historical evaluations. The original
  alpha/rho sweep requires a current legacy-all-prompts counterpart with usable
  state; it can retire an old checkpoint ahead of that counterpart.
- Rebuildable `merged/checkpoint-*` exports from current native runs. The exact
  step's native model shards, configuration and tokenizer must still be present.
  The normal pipeline rebuilds this export before evaluating.
- Older current checkpoints only after their complete five-dataset evaluation
  and a structurally complete newer resume point. Final weights only after the
  completion marker and all 20 scheduled five-dataset evaluations validate.
- `full_generation` text inside known Math evaluation JSON files. The filename,
  question, answer, per-sample correctness, format, token count, stop reason,
  evaluation settings and all numeric summaries remain. Text hashes and character
  counts replace the deleted text. No backup of that text is made.

The new run continues its own trajectory; deleting an ahead historical run does
not transfer its training progress to the new run.

## Keeps

- Every unfinished current job's latest checkpoint, including GRPO and OPSD,
  not just the ten alpha/rho sweep runs. Incomplete new saves retain older
  fallback checkpoints too. Latest-iteration pointers and run state remain.
- Native resume checks cover actor model, optimizer, scheduler/RNG shards,
  `data.pt`, tracker and, for SR-OPSD, all EMA teacher model shards.
- Registered eight-rank TRL/DeepSpeed resumes require their adapter/model,
  trainer state, latest tag, all eight optimizer and RNG shards, and DeepSpeed
  model state. These are structural checks, not a GPU restore test.
- The only copy needed for a pending evaluation, unless a same-step native
  source can rebuild it.
- All Physics trees, including raw logits, top-K probes, generations, aggregate
  tables and figures. The small downloaded analysis packs do not establish that
  the original full-vocabulary logits were backed up.
- All mcRL/MARQ/GPTQ projects, shared models, environments, datasets, code,
  metrics, logs and uploader state. Unregistered runs inside current roots are
  not touched.

Compacted Math files still pass the existing evaluator-completion and W&B header
readers. File fingerprints change, so an uploader may notice them again, but the
metric values do not change. Do not reset uploader state. Reading response text,
new text-based grading, Distinct-n, or qualitative examples is no longer possible
from the compacted files. Correctness/answer/length analyses remain possible.

## Development Machine Command

Stop custom jobs using these experiment paths before applying; do not schedule a
new launch during cleanup. Standard nightly and pipeline locks are honored. Files
changed within five minutes are skipped. A local PID check cannot determine
whether another machine is using the shared files. Close any older cleanup job
before starting this one. The script never kills training or uploader processes.

Run this block in the development machine's terminal. It runs the cleanup in a
child shell, so a command failure does not exit the interactive terminal.

```bash
bash <<'BASH'
set -euo pipefail
REPO=/media/damoxing/che-liu-fileset/ylong/sdpo/code/distIL-sr-opsd-renyi
BRANCH=codex/sr-opsd-renyi
OK=0
for attempt in 1 2 3; do
  echo "Fetch attempt $attempt/3"
  if timeout 90 git -C "$REPO" -c http.version=HTTP/1.1 fetch origin "$BRANCH"; then
    OK=1
    break
  fi
  sleep 5
done
[[ "$OK" == 1 ]] || { echo 'Fetch failed; nothing deleted'; exit 1; }
git -C "$REPO" merge --ff-only "origin/$BRANCH"
SCRIPT="$REPO/scripts/maintenance/cleanup_opsd_extreme.py"
test -f "$SCRIPT"
mkdir -p "$HOME/workspace"
JOB=$(mktemp -d "$HOME/workspace/opsd_extreme_XXXXXX")
nohup python3 -u "$SCRIPT" \
  --root /media/vlm-ckp-fileset/ylong \
  --report-dir "$JOB/receipts" \
  --retire-historical --apply \
  >"$JOB/cleanup.log" 2>&1 </dev/null &
echo "PID=$!"
echo "LOG=$JOB/cleanup.log"
echo "RECEIPTS=$JOB/receipts"
tail -f "$JOB/cleanup.log"
BASH
```

Ctrl-C stops `tail`, not the background cleanup. For a dry run, remove `--apply`.
Reports include inventories, retained small weight metadata, before/after hashes,
an event journal and `summary.json`; they cannot restore deleted tensors or text.
`selected_weight_bytes` is selected allocated size, not a measured disk-space
delta. Compare `du` before/after for actual usage. Skipped or invalid files are
reported with a nonzero exit; do not bypass the checks to force deletion.

## Storage Floor With The Current Format

Previous server measurements, not a new live scan:

| Current alpha/rho sweep | Latest checkpoint each | Five checkpoints |
| --- | ---: | ---: |
| Qwen3-8B | 64.32 GiB | 321.60 GiB |
| Qwen3-4B | 35.38 GiB | 176.90 GiB |
| Total | | 498.50 GiB |

This is about 535 decimal GB, excluding shared base models, environments, results
and other current jobs. A save needs temporary space for the new checkpoint while
the old one remains; evaluation exports also consume temporary space. Do not set
a running storage quota to exactly the retained-checkpoint total.

Smaller storage may be possible by representing frozen base weights once and
saving the changing student/teacher adapters plus optimizer and progress state.
That requires a different, validated save/restore path. This cleanup does not
implement it, remove pieces from a retained checkpoint, or promise a size for it.

## Local Verification

```bash
python3 -m unittest discover -s scripts/maintenance -p 'test_*.py' -q
```

114 tests pass, including the actual existing evaluation validator and W&B header
reader against compacted synthetic results. No production checkpoint restore or
remote filesystem deletion was performed during development.
