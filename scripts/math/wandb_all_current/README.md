# Current Math uploads

`manage_current_math_wandb.sh` uploads to
`wenxuan-yuan-imperial-college-london/SDPO_math_test` by default.
It only reads experiment outputs; it does not train, evaluate, or delete checkpoints.

## Inventory

The manager covers 28 configured runs, not necessarily 28 started runs:

| Group | Runs | Source |
| --- | ---: | --- |
| Main and follow-up GRPO/OPSD | 13 | Original eight runs, original grouped 8B OPSD, and four current follow-ups |
| Original 8B alpha/rho sweep | 5 | `sr_opsd_math_alpha_rho_sweep_20260819` |
| Legacy-all-prompts 8B alpha/rho sweep | 5 | `sr_opsd_math_alpha_rho_sweep_legacy_allprompts_20260825` |
| 4B A800 alpha/rho sweep | 5 | `sr_opsd_math_4b_alpha_rho_sweep_eval5_n16_a800_20260829` |

Source roots are under `/media/vlm-ckp-fileset/ylong` by default.
The 14 current follow-ups consist of the two five-run sweeps above and:

- GRPO 4B/8B: `math_grpo_{4b,8b}_opsd_trl_aligned_eval5_n16_{a800,h200}_20260827`.
- OPSD grouped 4B: `math_4b_opsd_grouped8x8_eval5_n16_a800_20260827`.
- OPSD grouped 8B: `math_opsd_grouped8x8_eval5_n16_h200_legacy_allprompts_20260825`.

These do not replace the older native-VERL GRPO or chunked-evaluation runs.
Physics runs are not included. Missing directories are reported as waiting/not started.

## Running

Run on the development server with access to the shared output directories:

```bash
REPO=/media/damoxing/che-liu-fileset/ylong/sdpo/code/distIL-sr-opsd-renyi
MANAGER="$REPO/scripts/math/wandb_all_current/manage_current_math_wandb.sh"
bash "$MANAGER" doctor
bash "$MANAGER" progress
bash "$MANAGER" once
```

For the existing periodic uploader, restart it after a code update to load the
new inventory. It starts an upload cycle immediately, then waits 600 seconds
between cycles by default:

```bash
UPLOAD_INTERVAL_SECONDS=600 bash "$MANAGER" restart
bash "$MANAGER" status
```

To upload only the five Qwen3-4B A800 alpha/rho runs, use the dedicated
incremental wrapper. Re-running the command uploads only new training steps or
newly completed evaluation files and resumes the same five W&B run IDs:

```bash
UPLOADER="$REPO/scripts/math/sweeps/sr_opsd_alpha_rho_4b_a800/upload_alpha_rho_4b_sweep_to_wandb.sh"
bash "$UPLOADER" --dry-run
bash "$UPLOADER"
```

Use only one uploader against a given state directory at a time. Do not run
`once` concurrently with the watcher or launch another watcher on a different
development server. API credentials are read from `WANDB_API_KEY` or
`/root/.config/wandb/upload.env`; never put the key in commands or version control.

The 4B sweep source and state can be overridden with `SWEEP_4B_OUTPUT_ROOT` and
`SWEEP_4B_STATE_ROOT`. Its default state is
`/media/vlm-ckp-fileset/ylong/sdpo_math_test_current_upload_state/alpha_rho_4b_a800`.
Keep existing state directories: they track uploaded steps and evaluation files.
All previously configured run IDs and state locations remain unchanged.

`progress` is a read-only preview of pending metrics, not proof that they reached
W&B. Dataset metrics are uploaded as each complete JSON arrives. Pooled metrics
are uploaded only once all five datasets for that checkpoint pass validation.
Select `eval/step` as the evaluation chart's x-axis. A watcher cycle reporting
`failures=0` and `current Math upload cycle completed` is the local success signal;
check the project for the corresponding run and checkpoint metrics afterward.
