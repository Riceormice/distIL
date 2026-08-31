import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import cleanup_verified_physics_weights as cleanup


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def make_run(root, relative=cleanup.ALLOWED_RUNS[0]):
    run = root / relative
    experiment = root / relative.split("/")[0]
    write_json(experiment / "protocol/probe_manifest.json", {
        "probe_id": "test-probe", "protocol": {
            "schedule": {"total_steps": 420}, "capture": {"capture_freq": 5}}})
    write_json(run / "launch_config.json", {
        "total_steps": 420, "n_gpus_per_node": 2, "nnodes": 1})
    write_json(run / "state/training.complete", {
        "probe_id": "test-probe", "total_training_steps": 420,
        "expected_capture_steps": cleanup.EXPECTED_STEPS, "missing_capture_steps": []})
    write_json(run / "state/run.complete", {
        "probe_id": "test-probe", "expected_steps": cleanup.EXPECTED_STEPS})
    for step in cleanup.EXPECTED_STEPS:
        tag = f"step_{step:04d}"
        for kind in ("capture", "generation"):
            write_json(run / f"state/{kind}/{tag}.complete", {
                "probe_id": "test-probe", "step": step})
        files = [run / f"generation/{tag}.jsonl", run / f"evaluation/{tag}.json",
                 run / f"token_stats/{tag}.parquet"]
        for kind in ("topk_probe", "raw_logits_audit"):
            if kind == "raw_logits_audit" and step % 20:
                continue
            files += [run / f"{kind}/{tag}/rank_{rank:04d}.npz" for rank in range(2)]
        for path in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"retained evidence")
    for name in ("audit_token_summary.parquet", "topk_ratio_stats.parquet"):
        path = experiment / "aggregate" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"aggregate")
    weight = run / "model_states/global_step_420/actor/model.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"authorized weights")
    return run


class VerifiedCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "data"
        self.root.mkdir()
        self.report = self.base / "receipt"
        self.report.mkdir()

    def execute(self, apply):
        journal = self.report / "events.jsonl"
        with journal.open("w") as stream:
            return cleanup.clean_one(self.root, cleanup.ALLOWED_RUNS[0],
                                     self.report, stream, apply)

    def test_apply_deletes_only_weights_and_is_idempotent(self):
        run = make_run(self.root)
        outside = self.root / "math_exp/checkpoints/model.pt"
        outside.parent.mkdir(parents=True)
        outside.write_bytes(b"do not delete")
        before = {p: p.read_bytes() for p in run.rglob("*")
                  if p.is_file() and "model_states" not in p.parts}
        with patch.object(cleanup.time, "time", return_value=time.time() + 1000):
            self.assertEqual(self.execute(True), "DELETED")
            self.assertEqual(self.execute(True), "ABSENT")
        self.assertFalse((run / "model_states").exists())
        self.assertTrue(outside.exists())
        self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
        receipt = next(self.report.glob("*.json"))
        self.assertIn("weight_inventory", json.loads(receipt.read_text()))

    def test_dry_run_never_deletes(self):
        run = make_run(self.root)
        with patch.object(cleanup.time, "time", return_value=time.time() + 1000):
            self.assertEqual(self.execute(False), "DRY_RUN")
        self.assertTrue((run / "model_states/global_step_420/actor/model.pt").exists())

    def test_missing_completion_preserves_weights(self):
        run = make_run(self.root)
        (run / "state/run.complete").unlink()
        with self.assertRaises(FileNotFoundError):
            self.execute(True)
        self.assertTrue((run / "model_states").exists())

    def test_wrong_probe_or_missing_shard_blocks_deletion(self):
        run = make_run(self.root)
        marker = run / "state/capture/step_0420.complete"
        write_json(marker, {"probe_id": "different-probe", "step": 420})
        with self.assertRaises(cleanup.UnsafeCleanup):
            self.execute(True)
        write_json(marker, {"probe_id": "test-probe", "step": 420})
        (run / "topk_probe/step_0420/rank_0001.npz").unlink()
        with self.assertRaises(cleanup.UnsafeCleanup):
            self.execute(True)
        self.assertTrue((run / "model_states").exists())

    def test_partial_training_blocks_deletion(self):
        run = make_run(self.root)
        path = run / "state/training.complete"
        data = json.loads(path.read_text())
        data["missing_capture_steps"] = [420]
        write_json(path, data)
        with self.assertRaises(cleanup.UnsafeCleanup):
            self.execute(True)
        self.assertTrue((run / "model_states").exists())

    def test_missing_generation_or_aggregate_blocks_deletion(self):
        run = make_run(self.root)
        (run / "generation/step_0005.jsonl").unlink()
        with self.assertRaises(FileNotFoundError):
            self.execute(True)
        (run / "generation/step_0005.jsonl").write_text("restored")
        experiment = self.root / cleanup.ALLOWED_RUNS[0].split("/")[0]
        (experiment / "aggregate/topk_ratio_stats.parquet").unlink()
        with self.assertRaises(FileNotFoundError):
            self.execute(True)
        self.assertTrue((run / "model_states").exists())

    def test_unreviewed_runs_and_symlinks_rejected(self):
        with self.assertRaises(cleanup.UnsafeCleanup):
            cleanup.clean_one(self.root, "math_exp", self.report, io.StringIO(), True)
        run = make_run(self.root)
        external = self.base / "external"
        external.mkdir()
        target = run / "model_states"
        target.rename(run / "saved-model_states")
        target.symlink_to(external, target_is_directory=True)
        with self.assertRaises(cleanup.UnsafeCleanup):
            self.execute(True)
        self.assertTrue(external.is_dir())

    def test_launcher_lock_blocks_deletion(self):
        run = make_run(self.root)
        with (run / "state/pipeline.lock").open("w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(cleanup.UnsafeCleanup):
                self.execute(True)
        self.assertTrue((run / "model_states").exists())

    def test_nightly_fkl_lock_blocks_deletion(self):
        relative = cleanup.ALLOWED_RUNS[3]
        run = make_run(self.root, relative)
        path = self.root / "nightly_experiment_state/physics_logits_sdpo_fkl/nightly.lock"
        path.parent.mkdir(parents=True)
        with path.open("w") as lock, (self.report / "events.jsonl").open("w") as stream:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(cleanup.UnsafeCleanup):
                cleanup.clean_one(self.root, relative, self.report, stream, True)
        self.assertTrue((run / "model_states").exists())

    def test_recent_or_contaminated_weights_rejected(self):
        run = make_run(self.root)
        with self.assertRaises(cleanup.UnsafeCleanup):
            cleanup.weight_inventory(self.root, run / "model_states")
        (run / "model_states/generation").mkdir()
        with self.assertRaises(cleanup.UnsafeCleanup):
            cleanup.weight_inventory(self.root, run / "model_states", quiet_seconds=0)

    def test_symlink_within_weights_is_rejected(self):
        run = make_run(self.root)
        external = self.base / "raw-evidence"
        external.write_bytes(b"retain")
        (run / "model_states/alias.pt").symlink_to(external)
        with self.assertRaises(cleanup.UnsafeCleanup):
            cleanup.weight_inventory(self.root, run / "model_states", quiet_seconds=0)
        self.assertEqual(external.read_bytes(), b"retain")

    def test_mutation_during_checks_blocks_deletion(self):
        run = make_run(self.root)
        original = cleanup.weight_inventory
        calls = 0

        def changing_inventory(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                (run / "model_states/new.pt").write_bytes(b"writer active")
            return original(*args, **kwargs)

        with patch.object(cleanup, "weight_inventory", side_effect=changing_inventory):
            with patch.object(cleanup.time, "time", return_value=time.time() + 1000):
                with self.assertRaises(cleanup.UnsafeCleanup):
                    self.execute(True)
        self.assertTrue((run / "model_states/global_step_420/actor/model.pt").exists())

    def test_cli_rejects_report_inside_data_root(self):
        result = subprocess.run([
            sys.executable, "-B", cleanup.__file__, "--root", str(self.root),
            "--report-dir", str(self.root / "reports"), "--apply",
        ], text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("outside", result.stderr)


if __name__ == "__main__":
    unittest.main()
