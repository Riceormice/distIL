import fcntl
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cleanup_reviewed_math_weights as cleanup


class ReviewedMathCleanupTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / "data"
        self.root.mkdir()
        self.report = self.base / "receipt"
        self.report.mkdir()
        self.candidate = next(c for c in cleanup.CANDIDATES if "math_grpo_8b_native" in c.path)
        clock = patch.object(cleanup.time, "time", return_value=time.time() + 1000)
        clock.start()
        self.addCleanup(clock.stop)

    def make_target(self, candidate=None):
        candidate = candidate or self.candidate
        target = self.root / candidate.path
        (target / "actor").mkdir(parents=True)
        (target / "actor/model.pt").write_bytes(b"retired tensors")
        (target / "config.json").write_text('{"model_type": "qwen3"}')
        owner = self.root / candidate.owner
        (owner / "state").mkdir(exist_ok=True)
        (owner / "evaluations/checkpoint-5").mkdir(parents=True, exist_ok=True)
        (owner / "evaluations/checkpoint-5/aime24.json").write_text('{"acc": 0.1}')
        (owner / "logs").mkdir(exist_ok=True)
        (owner / "logs/pipeline.log").write_text("retained training log")
        return target

    def execute(self, apply=True, candidate=None):
        with (self.report / "events.jsonl").open("a") as stream:
            return cleanup.clean_one(self.root, candidate or self.candidate, self.report, stream, apply)

    def make_current(self, candidate):
        run = self.root / candidate.current_run
        checkpoint = run / "native/checkpoints" / run.name / f"global_step_{candidate.minimum_step}"
        (checkpoint / "actor").mkdir(parents=True)
        (checkpoint / "data.pt").write_bytes(b"data")
        for group in ("model", "optim", "extra_state"):
            for rank in range(8):
                (checkpoint / f"actor/{group}_world_size_8_rank_{rank}.pt").write_bytes(b"state")
        return checkpoint

    def test_manifest_is_fixed_and_excludes_held_and_current_runs(self):
        self.assertEqual(len(cleanup.CANDIDATES), 24)
        self.assertEqual(len({c.path for c in cleanup.CANDIDATES}), 24)
        self.assertEqual(len({Path(c.path).parts[0] for c in cleanup.CANDIDATES}), 7)
        for c in cleanup.CANDIDATES:
            self.assertFalse(any(c.path.startswith(h + "/") for h in cleanup.HELD_RUNS))
            self.assertNotIn("legacy_allprompts", c.path)
            self.assertNotIn("opsd_trl_aligned", c.path)
            self.assertNotIn("physics", c.path)
            self.assertIsNotNone(cleanup.STEP.fullmatch(Path(c.path).name))

    def test_apply_only_removes_selected_weights_and_preserves_metadata_receipt(self):
        target = self.make_target()
        protected = []
        for path in ("other_project/checkpoints/a.pt", "physics/topk_probe/a.npz",
                     "current/evaluations/a.json", "current/checkpoints/a.pt"):
            p = self.root / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"protected")
            protected.append(p)
        owner = self.root / self.candidate.owner
        before = {p: p.read_bytes() for p in owner.rglob("*") if p.is_file() and target not in p.parents}
        self.assertEqual(self.execute()[0], "DELETED")
        self.assertFalse(target.exists())
        self.assertTrue(all(p.read_bytes() == b"protected" for p in protected))
        self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
        receipts = list(self.report.glob("*/inventory.json"))
        self.assertEqual(len(receipts), 1)
        self.assertIn("retained_evidence", json.loads(receipts[0].read_text()))
        config = receipts[0].parent / "checkpoint_metadata/config.json"
        self.assertEqual(json.loads(config.read_text())["model_type"], "qwen3")
        self.assertEqual(self.execute()[0], "ABSENT")

    def test_dry_run_never_deletes(self):
        target = self.make_target()
        self.assertEqual(self.execute(False)[0], "WOULD_DELETE")
        self.assertTrue(target.exists())

    def test_unknown_candidate_is_never_authorized(self):
        c = cleanup.Candidate("other_project/checkpoints/checkpoint-5", "other_project", "untrusted")
        with self.assertRaisesRegex(cleanup.UnsafeCleanup, "24 reviewed"):
            self.execute(candidate=c)

    def test_reclassified_current_root_blocks_deletion(self):
        target = self.make_target()
        with patch.object(cleanup, "scopes", return_value={target.relative_to(self.root).parts[0]: "current"}):
            with self.assertRaisesRegex(cleanup.UnsafeCleanup, "not historical"):
                self.execute()
        self.assertTrue(target.exists())

    def test_new_checkpoint_in_old_run_blocks_deletion(self):
        target = self.make_target()
        (target.parent / "global_step_15").mkdir()
        with self.assertRaisesRegex(cleanup.UnsafeCleanup, "unreviewed checkpoint"):
            self.execute()
        self.assertTrue(target.exists())

    def test_busy_pipeline_lock_blocks(self):
        target = self.make_target()
        lock = self.root / self.candidate.owner / "state/pipeline.lock"
        with lock.open("w") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(cleanup.UnsafeCleanup, "lock is held"):
                self.execute()
        self.assertTrue(target.exists())

    def test_symlink_target_is_rejected(self):
        target = self.make_target()
        moved = target.with_name("moved")
        target.rename(moved)
        target.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(cleanup.UnsafeCleanup, "symlink"):
            self.execute()
        self.assertTrue((moved / "actor/model.pt").exists())

    def test_symlink_inside_target_is_rejected(self):
        target = self.make_target()
        external = self.base / "keep.pt"
        external.write_bytes(b"keep")
        (target / "linked.pt").symlink_to(external)
        with self.assertRaisesRegex(cleanup.UnsafeCleanup, "symlink"):
            self.execute()
        self.assertEqual(external.read_bytes(), b"keep")

    def test_nested_mount_is_rejected(self):
        target = self.make_target()
        real = cleanup.os.path.ismount
        with patch.object(cleanup.os.path, "ismount", side_effect=lambda p: Path(p) == target or real(p)):
            with self.assertRaisesRegex(cleanup.UnsafeCleanup, "nested mount"):
                self.execute()

    def test_special_file_is_rejected(self):
        target = self.make_target()
        os.mkfifo(target / "fifo")
        with self.assertRaisesRegex(cleanup.UnsafeCleanup, "special file"):
            self.execute()

    def test_evidence_inside_weights_blocks(self):
        target = self.make_target()
        (target / "evaluations").mkdir()
        with self.assertRaisesRegex(cleanup.UnsafeCleanup, "evidence/runtime"):
            self.execute()
        self.assertTrue(target.exists())

    def test_recent_weight_write_blocks(self):
        target = self.make_target()
        with patch.object(cleanup.time, "time", return_value=target.stat().st_ctime + 1):
            with self.assertRaisesRegex(cleanup.UnsafeCleanup, "recently changed"):
                self.execute()

    def test_recent_evaluation_write_blocks_without_a_lock(self):
        target = self.make_target()
        value = time.time() + 5000
        output = self.root / self.candidate.owner / "evaluations/checkpoint-5/aime24.json"
        os.utime(output, (value, value))
        with self.assertRaisesRegex(cleanup.UnsafeCleanup, "recently changed"):
            self.execute()
        self.assertTrue(target.exists())

    def test_missing_current_resume_preserves_old_sweep(self):
        candidate = cleanup.CANDIDATES[0]
        target = self.make_target(candidate)
        with self.assertRaisesRegex(cleanup.UnsafeCleanup, "current checkpoint"):
            self.execute(candidate=candidate)
        self.assertTrue(target.exists())

    def test_current_sweep_is_retained_and_all_shards_required(self):
        candidate = cleanup.CANDIDATES[0]
        target = self.make_target(candidate)
        current = self.make_current(candidate)
        shard = current / "actor/optim_world_size_8_rank_7.pt"
        shard.unlink()
        with self.assertRaisesRegex(cleanup.UnsafeCleanup, "incomplete"):
            self.execute(candidate=candidate)
        shard.write_bytes(b"state")
        self.assertEqual(self.execute(candidate=candidate)[0], "DELETED")
        self.assertFalse(target.exists())
        self.assertTrue(shard.exists())

    def test_current_checkpoint_below_reviewed_minimum_is_not_sufficient(self):
        candidate = cleanup.CANDIDATES[0]
        target = self.make_target(candidate)
        current = self.make_current(candidate)
        current.rename(current.with_name("global_step_5"))
        with self.assertRaisesRegex(cleanup.UnsafeCleanup, "current checkpoint"):
            self.execute(candidate=candidate)
        self.assertTrue(target.exists())

    def test_modification_during_verification_blocks(self):
        target = self.make_target()
        original = cleanup.save_receipt

        def mutate(*args, **kwargs):
            receipt = original(*args, **kwargs)
            (target / "actor/model.pt").write_bytes(b"concurrent update")
            return receipt

        with patch.object(cleanup, "save_receipt", side_effect=mutate):
            with self.assertRaisesRegex(cleanup.UnsafeCleanup, "changed during verification"):
                self.execute()
        self.assertTrue(target.exists())

    def test_apply_requires_explicit_retirement(self):
        with self.assertRaises(SystemExit) as error, patch("sys.stderr", new=io.StringIO()):
            cleanup.main(["--root", str(self.root), "--report-dir", str(self.base / "new"), "--apply"])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.base / "new").exists())

    def test_report_cannot_be_in_experiment_root(self):
        with self.assertRaises(SystemExit), patch("sys.stderr", new=io.StringIO()):
            cleanup.main(["--root", str(self.root), "--report-dir", str(self.root / "report")])

    def test_complete_cli_applies_all_24_but_not_held_or_current(self):
        current_runs = set()
        kept_files = []
        for candidate in cleanup.CANDIDATES:
            self.make_target(candidate)
            if candidate.current_run and candidate.current_run not in current_runs:
                current_runs.add(candidate.current_run)
                kept_files.append(self.make_current(candidate) / "data.pt")
        for owner in cleanup.HELD_RUNS:
            path = self.root / owner / "native/checkpoints/held/global_step_65/actor/model.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"keep ahead checkpoint")
            kept_files.append(path)
        contents = {p: p.read_bytes() for p in kept_files}
        output = io.StringIO()
        report = self.base / "cli_report"
        with patch("sys.stdout", new=output):
            code = cleanup.main(["--root", str(self.root), "--report-dir", str(report),
                                 "--apply", "--retire-listed-historical-runs"])
        self.assertEqual(code, 0, output.getvalue())
        self.assertIn("finished deleted=24 failures_or_skips=0", output.getvalue())
        self.assertTrue(all(not (self.root / c.path).exists() for c in cleanup.CANDIDATES))
        self.assertTrue(all(p.read_bytes() == value for p, value in contents.items()))
        records = [json.loads(line) for line in (report / "events.jsonl").read_text().splitlines()]
        self.assertEqual(sum(row["status"] == "DELETED" for row in records), 24)

    def test_deleting_hardlink_does_not_remove_outside_copy(self):
        target = self.make_target()
        outside = self.root / "current_weight.pt"
        os.link(target / "actor/model.pt", outside)
        self.execute()
        self.assertEqual(outside.read_bytes(), b"retired tensors")


if __name__ == "__main__":
    unittest.main()
