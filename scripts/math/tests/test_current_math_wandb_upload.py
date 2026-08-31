#!/usr/bin/env python3
"""Regression checks for the current Math W&B upload inventory."""

from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


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
        self.assertIn("GRPO-OPSDTRLAligned", {item.method_label for item in specs})
        self.assertIn(
            "OPSD-Grouped8x8-LegacyAllPrompts",
            {item.method_label for item in specs},
        )

    def test_new_inventory_contains_fourteen_follow_up_runs(self) -> None:
        roots = [Path(f"/tmp/math-wandb-root-{index}") for index in range(7)]
        main_specs = common.run_specs(*roots)
        expected_profiles = {
            "8b-grpo-opsdtrl-aligned-h200-20260827",
            "4b-grpo-opsdtrl-aligned-a800-20260827",
            "8b-opsd-grouped8x8-legacyallprompts-h200-20260825",
            "4b-opsd-grouped8x8-a800-20260827",
        }
        new_main = [item for item in main_specs if item.profile in expected_profiles]

        self.assertEqual(len(new_main), 4)
        self.assertEqual(len(sweep.GRID), 5)
        self.assertEqual(len(new_main) + 2 * len(sweep.GRID), 14)

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

    def test_all_28_identities_are_distinct(self) -> None:
        roots = [Path(f"/tmp/math-wandb-root-{index}") for index in range(7)]
        ids = [common.run_id_for(item) for item in common.run_specs(*roots)]
        profiles = []
        for model, hardware, variant in (
            ("8B", "H200", "original"),
            ("8B", "H200", "legacy_allprompts"),
            ("4B", "A800", "legacy_allprompts"),
        ):
            for alpha, rho in sweep.GRID:
                item = sweep.SweepSpec(
                    roots[0], alpha, rho, variant=variant,
                    model_size=model, hardware=hardware,
                )
                ids.append(sweep.run_id_for(item))
                profiles.append(item.profile)
        self.assertEqual(len(ids), 28)
        self.assertEqual(len(set(ids)), 28)
        self.assertEqual(len(set(profiles)), 15)

    def test_4b_run_path_and_metadata_match_launcher(self) -> None:
        root = Path(
            "/media/vlm-ckp-fileset/ylong/"
            "sr_opsd_math_4b_alpha_rho_sweep_eval5_n16_a800_20260829"
        )
        item = sweep.SweepSpec(
            root, "0.9", "0.7", variant="legacy_allprompts",
            display_suffix="LegacyAllPrompts", model_size="4B", hardware="A800",
        )
        expected = (
            "sr-opsd-4b-seed0-native-verl-forward-renyi-"
            "rho0.7-refw0.9-sync0-ema0.05-lr5e-6-"
            "trainbs8-mbs8-rolloutn8-topk100-tailFalse-clip0.05-"
            "temp0.7-tok16384-steps100-sched420-eval5-n16-a800"
        )
        self.assertEqual(item.run_root, root / "sr_opsd" / expected)
        self.assertEqual(
            item.display_name,
            "Qwen3-4B-Math-SR-OPSD-alpha0.9-rho0.7-seed0-eval5-N16-"
            "A800-LegacyAllPrompts",
        )
        config = sweep.config_for(item, {})
        self.assertEqual(config["model"], "Qwen3-4B")
        self.assertEqual(config["hardware"], "8xA800")
        self.assertEqual(config["source_root"], str(item.run_root))
        self.assertEqual(config["evaluation_variant"], "legacy_allprompts")
        self.assertEqual(config["self_reference_alpha"], 0.9)
        self.assertEqual(config["renyi_order_rho"], 0.7)


class FourBSweepUploadTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.spec = sweep.SweepSpec(
            self.root / "outputs", "0.7", "0.95",
            variant="legacy_allprompts", model_size="4B", hardware="A800",
        )
        self.state_dir = self.root / "state"
        metrics = self.spec.run_root / "native/logs" / self.spec.run_name / "metrics.jsonl"
        metrics.parent.mkdir(parents=True)
        metrics.write_text(json.dumps({"step": 5, "data": {"loss": 0.125}}) + "\n")

    def write_eval(self, dataset: str, *, step: int = 5) -> Path:
        problems = common.EXPECTED_PROBLEMS[dataset]
        total = problems * common.VAL_N
        payload = {
            "dataset": dataset, "num_problems": problems,
            "val_n": common.VAL_N, "total_solutions": total,
            "average_at_n": total, "pass_at_n": problems,
            "majority_vote_at_n": problems, "formatted_count": total,
            "average_at_n_pct": 100.0, "pass_at_n_pct": 100.0,
            "majority_vote_at_n_pct": 100.0, "format_rate": 100.0,
            "results": [
                {"val_n": common.VAL_N,
                 "generations": [{"correct": True}] * common.VAL_N}
                for _ in range(problems)
            ],
        }
        path = self.spec.run_root / "evaluations" / f"checkpoint-{step}" / f"{dataset}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return path

    def upload(self, *, dry_run: bool = False) -> tuple[int, int]:
        with contextlib.redirect_stdout(io.StringIO()):
            return sweep.upload_spec(
                self.spec, entity="test-entity", project="test-project",
                state_dir=self.state_dir, dry_run=dry_run, skip_validation=False,
            )

    def test_dry_run_reads_real_layout_without_writing_state(self) -> None:
        for dataset in common.DATASETS:
            self.write_eval(dataset)
        self.assertEqual(self.upload(dry_run=True), (7, 0))
        self.assertFalse(self.state_dir.exists())

    def test_partial_eval_upload_then_resume_only_new_data(self) -> None:
        for dataset in common.DATASETS[:-1]:
            self.write_eval(dataset)
        wandb = MagicMock()
        run = wandb.init.return_value
        run.summary = {}
        with patch.dict(sys.modules, {"wandb": wandb}):
            self.assertEqual(self.upload(), (5, 0))
            config = wandb.init.call_args.kwargs
            self.assertEqual(config["id"], sweep.run_id_for(self.spec))
            self.assertEqual(config["resume"], "allow")
            self.assertEqual(config["config"]["model"], "Qwen3-4B")
            self.assertIn("A800", config["tags"])
            self.assertIn("Qwen3-4B", config["group"])
            first_logs = [call.args[0] for call in run.log.call_args_list]
            self.assertFalse(any("eval/pooled/avg@16_pct" in row for row in first_logs))
            run.finish.assert_called_once()

            self.write_eval(common.DATASETS[-1])
            run.log.reset_mock()
            self.assertEqual(self.upload(), (2, 0))
            resumed_logs = [call.args[0] for call in run.log.call_args_list]
            pooled = [row for row in resumed_logs if "eval/pooled/avg@16_pct" in row]
            self.assertEqual(len(pooled), 1)
            self.assertEqual(pooled[0]["eval/step"], 5)
            self.assertEqual(pooled[0]["eval/pooled/avg@16_pct"], 100.0)
            self.assertEqual(run.summary["progress/uploaded_eval_files"], 5)

            wandb.init.reset_mock()
            self.assertEqual(self.upload(), (0, 0))
            wandb.init.assert_not_called()

    def test_incomplete_file_is_not_uploaded(self) -> None:
        path = self.write_eval("aime24")
        path.write_text('{"dataset": "aime24", "results": [')
        self.assertEqual(self.upload(dry_run=True), (1, 0))

    def test_cli_selects_all_five_4b_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SWEEP_PATH), "--output-root", str(self.spec.output_root),
             "--model-size", "4B", "--hardware", "A800",
             "--variant", "legacy_allprompts", "--state-dir", str(self.state_dir),
             "--dry-run"],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("dry-run: runs=5, events_selected=1, failures=0", result.stdout)
        self.assertEqual(result.stdout.count("Qwen3-4B-Math-SR-OPSD-"), 5)
        self.assertNotIn("Qwen3-8B", result.stdout)


class CurrentMathManagerTest(unittest.TestCase):
    def test_progress_and_once_include_fourth_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            math_dir = root / "math"
            manager = math_dir / "wandb_all_current/manage_current_math_wandb.sh"
            manager.parent.mkdir(parents=True)
            shutil.copyfile(REPO / "scripts/math/wandb_all_current" / manager.name, manager)
            stub = (
                '#!/usr/bin/env bash\n'
                'printf "%s|%s|%s|%s\\n" "$WANDB_PROJECT" '
                '"${OUTPUT_ROOT:-main}" "$STATE_ROOT" "$*" >> "$TEST_CALLS"\n'
            )
            for relative in (
                "wandb_math_8runs/manage_math_8runs_wandb.sh",
                "sweeps/sr_opsd_alpha_rho_8b_h200/upload_alpha_rho_sweep_to_wandb.sh",
            ):
                path = math_dir / relative
                path.parent.mkdir(parents=True)
                path.write_text(stub)
            env = {
                **os.environ,
                "STATE_ROOT": str(root / "state"),
                "MAIN_STATE_ROOT": str(root / "state/main"),
                "SWEEP_OUTPUT_ROOT": str(root / "old8b"),
                "SWEEP_STATE_ROOT": str(root / "state/old8b"),
                "LEGACY_SWEEP_OUTPUT_ROOT": str(root / "legacy8b"),
                "LEGACY_SWEEP_STATE_ROOT": str(root / "state/legacy8b"),
                "SWEEP_4B_OUTPUT_ROOT": str(root / "new4b"),
                "SWEEP_4B_STATE_ROOT": str(root / "state/new4b"),
                "WANDB_ENTITY": "test-entity", "WANDB_PROJECT": "test-project",
                "TEST_CALLS": str(root / "calls"),
            }
            for command in ("progress", "once"):
                with self.subTest(command=command):
                    Path(env["TEST_CALLS"]).write_text("")
                    result = subprocess.run(
                        ["bash", str(manager), command], env=env,
                        capture_output=True, text=True, check=True,
                    )
                    self.assertIn("managed_runs=28", result.stdout)
                    calls = Path(env["TEST_CALLS"]).read_text().splitlines()
                    self.assertEqual(len(calls), 4)
                    self.assertTrue(all(line.startswith("test-project|") for line in calls))
                    self.assertIn(str(root / "new4b"), calls[-1])
                    self.assertIn("--state-dir " + str(root / "state/new4b"), calls[-1])
                    self.assertIn("--model-size 4B --hardware A800", calls[-1])
                    self.assertIn("--variant legacy_allprompts", calls[-1])
                    self.assertEqual("--dry-run" in calls[-1], command == "progress")


if __name__ == "__main__":
    unittest.main()
