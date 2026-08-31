# Retire Old LCB Generated Responses

`slim_lcb17_results_archive.py` handles only the reviewed archive named
`sdpo_lcbv6_sr_opsd_1_7b_h200_results.tar.gz` (reported as 37G on the server).

With `--apply`, it permanently removes these archive subtrees:

- `logs/lcbv6-sr_opsd-Qwen3-1.7B-*/rollouts/`: training prompts/responses and per-sample records.
- `logs/lcbv6-sr_opsd-Qwen3-1.7B-*/validation/`: validation prompts/responses and per-sample records.

The LCB launcher assigns these directories to `trainer.rollout_data_dir` and
`trainer.validation_data_dir`; VERL `_dump_generations` writes the originals there.
The request authorizes discarding both. Existing `metrics.jsonl`, configs,
launcher/train logs, and all members outside those two subtrees remain byte-identical.
Console logs are retained verbatim and can contain incidental response excerpts.
It does not touch live experiment directories, Math resume checkpoints, Physics
logit captures, model assets, environments, or any other project.

## Run

```bash
python3 -u scripts/maintenance/slim_lcb17_results_archive.py \
  --archive /media/vlm-ckp-fileset/ylong/result_archives/sdpo_lcbv6_sr_opsd_1_7b_h200_results.tar.gz \
  --report-dir "$HOME/workspace/lcb17_slim_$(date +%Y%m%d_%H%M%S)" \
  --apply
```

Omit `--apply` for an audit. Both modes must read the entire compressed source.
No extraction or large uncompressed temporary copy is created. Only the retained
compressed members need temporary disk space. Progress is printed during scanning;
gzip input cannot skip response payloads without decompressing them.

The script checks source stability, source/replacement gzip integrity, retained
member content hashes and metadata, and a nonempty metrics file for every affected
run. It refuses unsafe paths, duplicates, links, sparse/special members, concurrent
cleanup, and a replacement that is not smaller. It never follows archive links.
Do not run other programs that rewrite this archive during cleanup.

After verification, an atomic rename replaces the **same archive path** with the
smaller archive. There is no separate backup of the old raw responses. The adjacent
`.sha256` file is updated. `ready.json` records the verified replacement manifest;
`complete.json` records a completed replacement. A failure before replacement leaves
the old archive intact. A crash after replacement cannot undo it; the verified
replacement checksum remains in `ready.json` even if the sidecar/complete receipt
was not written. A normal rerun with a fresh report directory reports `NO_CHANGE`.
If killed with SIGKILL during copying, remove only that invocation's `.slim.*.part`
after confirming it stopped; normal failures remove their own temporary file.

Reported member byte counts are uncompressed; before/after archive sizes are
compressed. Reclaimed filesystem space can differ due to hardlinks or snapshots.
The old per-sample responses will no longer be available for regrading or diversity
analysis. This script preserves existing metrics; it does not claim to reconstruct
or independently verify their scientific correctness.
