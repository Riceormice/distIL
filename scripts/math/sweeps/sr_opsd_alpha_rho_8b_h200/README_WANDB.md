# Upload the alpha/rho Math sweep to W&B

The uploader reads the five Qwen3-8B SR-OPSD runs under:

```text
/media/vlm-ckp-fileset/ylong/sr_opsd_math_alpha_rho_sweep_20260819
```

It validates N=16 evaluation JSON files, uploads all currently available
training/evaluation steps, and records local state so rerunning only uploads new
events.

```bash
SCRIPT=/media/damoxing/che-liu-fileset/ylong/sdpo/code/distIL-sr-opsd-renyi/scripts/math/sweeps/sr_opsd_alpha_rho_8b_h200/upload_alpha_rho_sweep_to_wandb.sh

bash "$SCRIPT" --dry-run
bash "$SCRIPT"
```

Destination by default:

```text
wenxuan-yuan-imperial-college-london/SDPO_math_test
```

The API key is loaded from `/root/.config/wandb/upload.env`; it is never stored
in this repository.
