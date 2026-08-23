# Missing Physics alpha-rho grid point

This launcher adds the missing Qwen3-8B Physics result with self-reference
coefficient `alpha=0.90` and Renyi order `rho=0.90`. It matches the earlier
grid: seed 0, 420 steps, validation every 5 steps with 16 samples, frozen
initial reference (`sync=0`), and no checkpoints.

```bash
exec bash /media/damoxing/che-liu-fileset/ylong/sdpo/code/distIL-sr-opsd-renyi/scripts/physics/alpha090_rho090_a800/a800_alpha090_rho090.sh
```
