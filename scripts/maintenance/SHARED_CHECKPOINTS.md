# Shared checkpoint storage

## What Changes

The ten current Math SR-OPSD sweeps now use lossless shared model storage.
Identical student/teacher tensor blocks are stored once per model architecture,
world size and rank. Each checkpoint keeps its own changes. Teacher EMA rounding
differences are preserved, not assumed to be zero. The first converted student
shard seeds an immutable baseline; it is not necessarily the original HF model.
The HF model assets are still required for model construction and evaluation.

Optimizer, scheduler, RNG, data-loader progress and checkpoint cadence are
unchanged. No precision conversion, quantization, optimizer reset, training
parameter change or evaluation change is made. Unfinished Math runs retain their
existing current-resume-point policy. The four OPSD/TRL GRPO and grouped-8x8 jobs
already save adapters; their existing adapter/DeepSpeed resume files are kept.

Physics full-parameter training remains full-parameter training. Its changed
weights cannot be discarded as though they were frozen. The P0 repository gets
the same lossless codec for future saves. Full-parameter savings may be smaller.

Shared files live under:

```text
/media/vlm-ckp-fileset/ylong/sdpo/shared_checkpoint_bases/v1
```

**Do not delete that directory while any compact checkpoint refers to it.**
Back up the shared directory together with checkpoint directories, or expand the
checkpoints first. The manifests store absolute shared-file paths. Copying only a
compact `.pt` to another machine is insufficient. Standard `torch.load(path)`
does not understand the new format; the updated VERL loader and merger do.

## Validation

The converter verifies SHA-256 of the entire reconstructed original Torch
archive before replacing each shard atomically. Dtype, tensor aliases, FSDP
metadata and every stored parameter byte are preserved. Old plain shards and
new compact shards may coexist, including after a partially completed migration.

Conversion only walks the current experiment registry. It does not scan mcRL,
MARQ, GPTQ, historical runs or raw generation/logits directories. It requires
`--jobs-paused`, respects nightly/pipeline locks and refuses files written in the
last five minutes. **Stop affected jobs and their scheduled starts first.** Locks
alone cannot prove that an old remote job is stopped.

Saving currently writes the normal shard and then compacts it. Allow temporary
space for that shard, its private payload and verified replacement; concurrent
ranks need this space simultaneously. The first save also creates shared bases.
This reduces retained storage, not training GPU memory, and adds checkpoint I/O
and CPU work. Savings must include both the run directories and shared directory;
no production size estimate has been substituted for measurement.

Local tests cover exact archive/tensor restoration, optimizer next-update
equality, distributed DTensor/ShardedTensor, teacher rounding differences,
conversion interruption/retry, symlink and missing-shard refusal, and expansion.
The real VERL FSDP checkpoint-manager smoke also passes in both repositories on
two CPU ranks for adapter and full-parameter cases, including byte-exact next
optimizer/teacher updates and identical merged weights for evaluation.
An eight-GPU smoke script is provided below; local CPU checks do not certify the
cluster's CUDA/FSDP/runtime combination.

## Development Machine: Update

These blocks run in child shells so a failed command does not close the terminal.
Do not run the conversion concurrently with a training/evaluation job or cleanup.

```bash
bash <<'BASH'
set -euo pipefail
REPO=/media/damoxing/che-liu-fileset/ylong/sdpo/code/distIL-sr-opsd-renyi
P0=/media/damoxing/che-liu-fileset/ylong/sdpo/code/SDPO-p0-mechanism
update_repo() {
  local repo="$1" branch="$2" ok=0
  test -z "$(git -C "$repo" status --porcelain)"
  test "$(git -C "$repo" branch --show-current)" = "$branch"
  for attempt in 1 2 3; do
    echo "Fetch $branch ($attempt/3)"
    if timeout 90 git -C "$repo" -c http.version=HTTP/1.1 fetch origin "$branch"; then
      ok=1
      break
    fi
    sleep 5
  done
  test "$ok" = 1
  git -C "$repo" merge --ff-only "origin/$branch"
  git -C "$repo" rev-parse --short HEAD
}
update_repo "$REPO" codex/sr-opsd-renyi
update_repo "$P0" codex/p0-mechanism-evidence-fixes
cmp "$REPO/SDPO/verl/utils/checkpoint/shared_model.py" "$P0/verl/utils/checkpoint/shared_model.py"
echo "CODE READY"
BASH
```

## Compute Machine: Small Eight-GPU Test

Run on an idle eight-GPU node using the existing unified environment. This uses
tiny randomly initialized models, not the experiment checkpoints. It tests frozen
base plus adapter and full-parameter state; each includes a separate EMA teacher.

```bash
bash <<'BASH'
set -euo pipefail
REPO=/media/damoxing/che-liu-fileset/ylong/sdpo/code/distIL-sr-opsd-renyi
PY=/media/damoxing/che-liu-fileset/ylong/sdpo/envs/math-verl-current/bin/python
P0=/media/damoxing/che-liu-fileset/ylong/sdpo/code/SDPO-p0-mechanism
OUT=/media/vlm-ckp-fileset/ylong/sdpo/shared_checkpoint_smoke_$(date +%Y%m%d_%H%M%S)
export PYTHONNOUSERSITE=1
unset PYTHONPATH
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$REPO/scripts/math/smoke_shared_fsdp_checkpoint.py" --out "$OUT/main"
cat "$OUT/main/PASS.json"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$REPO/scripts/math/smoke_shared_fsdp_checkpoint.py" --out "$OUT/p0" --verl-root "$P0"
cat "$OUT/p0/PASS.json"
BASH
```

The real cluster test must print `SHARED FSDP SMOKE: PASS` before bulk conversion.
Both tests use their own new output directories and leave production data alone.

## Development Machine: Convert Existing Checkpoints

After the smoke passes, keep all affected jobs stopped and wait five minutes.
This converts retained checkpoints in the 16-job current registry; missing or
already-cleaned paths are reported, not recreated. Already-adapter-based TRL jobs
are reported without rewriting them. It does not delete any whole checkpoint.

```bash
bash <<'BASH'
set -euo pipefail
REPO=/media/damoxing/che-liu-fileset/ylong/sdpo/code/distIL-sr-opsd-renyi
BASE=/media/vlm-ckp-fileset/ylong
JOB="$HOME/workspace/shared_checkpoint_conversion_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$JOB"
nohup python3 -u "$REPO/scripts/maintenance/compact_current_checkpoints.py" \
  --root "$BASE" --report-dir "$JOB/receipts" --apply --jobs-paused \
  >"$JOB/conversion.log" 2>&1 </dev/null &
echo "PID=$!"
echo "LOG=$JOB/conversion.log"
echo "RECEIPTS=$JOB/receipts/conversion.jsonl"
tail -f "$JOB/conversion.log"
BASH
```

`FINISHED failures=0` means every found, applicable run finished conversion.
`RUN_TOTAL` includes its full retained checkpoint files. `SHARED_STORE` is the
additional shared cost and must be counted once, not once per run. Use `du -sh`
on the experiment roots and the shared directory for final actual usage.
`Ctrl-C` while following the log only stops `tail`; it does not cancel `nohup`.

For inventory only, omit `--apply --jobs-paused`. For just one first trial, add
`--job math_alpha070_rho070`. Every invocation needs a new `--report-dir`.
After an interrupted conversion, rerun the same block with a new report directory.
Do not manually delete the shared store or checkpoint `.pt` files.

## Resume And Rollback

The Math nightly commands/job IDs and output roots are unchanged. Use the original
commands after conversion finishes, with the updated code. Do not run an old
checkout against a compact checkpoint. Setting `SDPO_SHARED_CHECKPOINT_STORE=`
disables compaction for future saves; reading compact state remains supported.

To return to ordinary standalone `.pt` files, run the converter with
`--mode expand --apply --jobs-paused` and a fresh report directory. Expansion
requires enough free space for full shards. It does not delete the shared store
because other runs may still depend on it. `--mode verify` checks full logical
hashes without rewriting model files.

### Physics Legacy Limitation

The archived P0 implementation did not save EMA teacher state. This patch adds
teacher saving and restores it when present. It cannot reconstruct missing
teacher history in old checkpoints. New code rejects an exact resume without
teacher state. `SDPO_ALLOW_LEGACY_TEACHER_RESET=1` explicitly allows the former
approximate reset; that is **not** equivalent continuation and is not enabled by
the commands above.

The completed FKL/JSD runs do not need rerunning or probe changes. Their numerical
results, probe manifests and aggregates are untouched. Their code/probe revision
checks remain strict. A new Physics run needs a probe built under its new code
revision and the existing two-step acceptance test; an unfinished old Physics run
must not bypass provenance checks just to load a smaller file.
