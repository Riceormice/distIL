import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

import cleanup_verified_physics_weights as cleanup


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def write_npz(path, rows=1, version=(1, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = repr({"descr": "<i4", "fortran_order": False, "shape": (rows,)}).encode("ascii")
    size_fmt = "<H" if version == (1, 0) else "<I"
    prefix_size = 8 + struct.calcsize(size_fmt)
    header += b" " * ((64 - (prefix_size + len(header) + 1) % 64) % 64) + b"\n"
    raw = b"\x93NUMPY" + bytes(version) + struct.pack(size_fmt, len(header)) + header
    raw += struct.pack("<" + "i" * rows, *range(rows))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("question_idx.npy", raw)


def make_run(root, relative=cleanup.ALLOWED_RUNS[0], world_size=2,
             topk_ranks=(0, 1), audit_ranks=(0, 1)):
    run = root / relative
    experiment = root / relative.split("/")[0]
    write_json(experiment / "protocol/probe_manifest.json", {
        "probe_id": "test-probe", "protocol": {
            "schedule": {"total_steps": 420}, "capture": {"capture_freq": 5, "audit_freq": 20}}})
    write_json(run / "launch_config.json", {
        "total_steps": 420, "n_gpus_per_node": world_size, "nnodes": 1})
    write_json(run / "state/training.complete", {
        "probe_id": "test-probe", "total_training_steps": 420,
        "expected_capture_steps": cleanup.EXPECTED_STEPS, "missing_capture_steps": []})
    write_json(run / "state/run.complete", {
        "probe_id": "test-probe", "expected_steps": cleanup.EXPECTED_STEPS})
    for step in cleanup.EXPECTED_STEPS:
        tag = f"step_{step:04d}"
        for kind in ("capture", "generation"):
            marker = {"probe_id": "test-probe", "step": step}
            if kind == "capture":
                marker.update({"audit": step % 20 == 0, "topk_rows": len(topk_ranks),
                               "audit_rows": len(audit_ranks) if step % 20 == 0 else 0})
            write_json(run / f"state/{kind}/{tag}.complete", marker)
        files = [run / f"generation/{tag}.jsonl", run / f"evaluation/{tag}.json",
                 run / f"token_stats/{tag}.parquet"]
        for kind in ("topk_probe", "raw_logits_audit"):
            if kind == "raw_logits_audit" and step % 20:
                continue
            ranks = topk_ranks if kind == "topk_probe" else audit_ranks
            files += [run / f"{kind}/{tag}/rank_{rank:04d}.npz" for rank in ranks]
        for path in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".npz":
                write_npz(path)
            else:
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
        original_marker = json.loads(marker.read_text())
        write_json(marker, {"probe_id": "different-probe", "step": 420})
        with self.assertRaises(cleanup.UnsafeCleanup):
            self.execute(True)
        write_json(marker, original_marker)
        (run / "topk_probe/step_0420/rank_0001.npz").unlink()
        with self.assertRaises(cleanup.UnsafeCleanup):
            self.execute(True)
        self.assertTrue((run / "model_states").exists())

    def test_sparse_nonconsecutive_ranks_can_delete_completed_weights(self):
        run = make_run(self.root, world_size=8, topk_ranks=(0, 3, 7), audit_ranks=(1, 6))
        with patch.object(cleanup.time, "time", return_value=time.time() + 1000):
            self.assertEqual(self.execute(True), "DELETED")
        self.assertFalse((run / "model_states").exists())
        self.assertTrue((run / "raw_logits_audit/step_0000/rank_0006.npz").exists())

    def test_sparse_shard_missing_or_incorrect_row_count_still_blocks(self):
        run = make_run(self.root, world_size=8, audit_ranks=(1, 6))
        shard = run / "raw_logits_audit/step_0000/rank_0006.npz"
        shard.unlink()
        with self.assertRaisesRegex(cleanup.UnsafeCleanup, "found 1, expected 2"):
            self.execute(True)
        write_npz(shard, rows=3)
        with self.assertRaisesRegex(cleanup.UnsafeCleanup, "found 4, expected 2"):
            self.execute(True)
        self.assertTrue((run / "model_states").exists())

    def test_missing_sample_count_is_not_silently_accepted(self):
        run = make_run(self.root)
        path = run / "state/capture/step_0000.complete"
        marker = json.loads(path.read_text())
        marker.pop("audit_rows")
        write_json(path, marker)
        with self.assertRaisesRegex(cleanup.UnsafeCleanup, "expected sample count"):
            self.execute(True)
        self.assertTrue((run / "model_states").exists())

    def test_unexpected_rank_and_corrupt_npz_are_rejected(self):
        run = make_run(self.root)
        directory = run / "raw_logits_audit/step_0000"
        write_npz(directory / "rank_9999.npz")
        with self.assertRaisesRegex(cleanup.UnsafeCleanup, "unexpected logits rank"):
            self.execute(True)
        (directory / "rank_9999.npz").unlink()
        (directory / "rank_0000.npz").write_bytes(b"broken zip")
        with self.assertRaisesRegex(cleanup.UnsafeCleanup, "invalid logits shard index"):
            self.execute(True)
        self.assertTrue((run / "model_states").exists())

    def test_index_reader_handles_npy_versions(self):
        path = self.root / "test.npz"
        for version in ((1, 0), (2, 0), (3, 0)):
            with self.subTest(version=version):
                write_npz(path, rows=7, version=version)
                self.assertEqual(cleanup.npz_row_count(self.root, path), 7)

    def test_index_reader_matches_real_numpy_exports_when_available(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional NumPy compatibility test")
        path = self.root / "numpy-export.npz"
        for dtype in (np.int32, np.int64):
            for writer in (np.savez, np.savez_compressed):
                with self.subTest(dtype=dtype, writer=writer.__name__):
                    writer(path, question_idx=np.arange(7, dtype=dtype),
                           st_logp=np.zeros((7, 3, 100), dtype=np.float16))
                    self.assertEqual(cleanup.npz_row_count(self.root, path), 7)

    def test_index_reader_rejects_empty_and_truncated_index(self):
        path = self.root / "test.npz"
        write_npz(path, rows=0)
        with self.assertRaises(cleanup.UnsafeCleanup):
            cleanup.npz_row_count(self.root, path)
        write_npz(path, rows=3)
        with zipfile.ZipFile(path) as archive:
            raw = archive.read("question_idx.npy")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("question_idx.npy", raw[:-1])
        with self.assertRaisesRegex(cleanup.UnsafeCleanup, "truncated"):
            cleanup.npz_row_count(self.root, path)

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
