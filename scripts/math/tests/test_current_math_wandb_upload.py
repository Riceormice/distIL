#!/usr/bin/env python3
"""Regression checks for the current Math W&B upload inventory."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
COMMON_DIR = REPO / "scripts/math/wandb_math_8runs"
SWEEP_PATH = (
    REPO
    / "scripts/math/sweeps/sr_opsd_alpha_rho_8b_h200"
    / "upload_alpha_rho_sweep_to_wandb.py"
)
sys.path.insert(0, str(COMMON_DIR))

import upload_math_8runs_to_wandb as common  # noqa: E402


def load_sweep_module():
    spec = importlib.util.spec_from_file_location("alpha_rho_uploader", SWEEP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SWEEP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sweep = load_sweep_module()


class CurrentMathInventoryTest(unittest.TestCase):
    def test_main_inventory_has_unique_profiles_and_names(self) -> None:
        roots = [Path(f"/tmp/math-wandb-root-{index}") for index in range(7)]
        specs = common.run_specs(*roots)

        self.assertEqual(len(specs), 13)
        self.assertEqual(len({item.profile for item in specs}), 13)
        self.assertEqual(len({item.display_name for item in specs}), 13)
        self.assertEqual(
            common.run_id_for(specs[8]),
            "m8823d34731cec56",
            "the previously uploaded grouped 8x8 run must keep its W&B ID",
        )

    def test_new_main_runs_have_distinct_ids(self) -> None:
        roots = [Path(f"/tmp/math-wandb-root-{index}") for index in range(7)]
        specs = common.run_specs(*roots)
        run_ids = [common.run_id_for(item) for item in specs]

        self.assertEqual(len(set(run_ids)), len(run_ids))
        self.assertIn("GRPO-NativeFixed", {item.method_label for item in specs})
        self.assertIn(
            "OPSD-Grouped8x8-LegacyAllPrompts",
            {item.method_label for item in specs},
        )

    def test_legacy_sweep_does_not_overwrite_original(self) -> None:
        root = Path("/tmp/math-alpha-rho")
        original = sweep.SweepSpec(root, "0.9", "0.7")
        legacy = sweep.SweepSpec(
            root,
            "0.9",
            "0.7",
            variant="legacy_allprompts",
            display_suffix="LegacyAllPrompts",
        )

        self.assertEqual(sweep.run_id_for(original), "ar383a648c15e20b")
        self.assertNotEqual(sweep.run_id_for(original), sweep.run_id_for(legacy))
        self.assertEqual(original.display_name.endswith("LegacyAllPrompts"), False)
        self.assertTrue(legacy.display_name.endswith("LegacyAllPrompts"))


if __name__ == "__main__":
    unittest.main()
