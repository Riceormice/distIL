# Unified mathematics environments

The mathematics pipelines use two complete, versioned environments under
`/media/damoxing/che-liu-fileset/ylong/sdpo/envs`:

- `math-verl-current`: GRPO, SDPO, and SR-OPSD using native VERL.
- `math-opsd-current`: OPSD training and distIL mathematics evaluation.

Each `current` path is an atomic symlink to a versioned environment. Package
files and compiled extensions live inside that versioned prefix. The launchers
do not add another environment's `site-packages` to `PYTHONPATH`.

Build both environments on one GPU machine:

```bash
bash scripts/math/unified_env/prepare_unified_math_envs.sh
```

The build marker is written only after `pip check`, package-origin checks,
shared-memory checks, a Ray worker check, and the optional GPU FlashAttention
smoke test pass. To replace an existing build, use a new `BUILD_ID`; do not
modify an active versioned environment in place.

The builder uses a complete local FlashAttention source tree when available.
If the configured path only contains an extracted package fragment, it installs
and compiles `flash-attn==2.8.3` from the package index instead. The build is
forced locally so it does not depend on GitHub release-wheel availability.
