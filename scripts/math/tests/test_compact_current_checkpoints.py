import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "maintenance"))
convert = importlib.import_module("compact_current_checkpoints")
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU Torch needed for checkpoint fixtures")
class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.job, self.parent = convert.audit.current_jobs(self.root)[0]
        self.run = self.parent / "sr-opsd-fixture-rho0.7-refw0.7-sync0"
        self.ckpt = self.run / "native/checkpoints" / self.run.name / "global_step_5"
        self.actor = self.ckpt / "actor"
        self.models = []
        for directory in (self.actor, self.actor / "ema_teacher"):
            (directory / "huggingface").mkdir(parents=True)
            (directory / "fsdp_config.json").write_text('{"world_size": 2}')
            (directory / "huggingface/config.json").write_text('{"model_type": "test", "hidden_size": 256}')
            for rank in range(2):
                path = directory / f"model_world_size_2_rank_{rank}.pt"
                torch.save({"weight": torch.arange(65536, dtype=torch.float32)}, path)
                self.models.append(path)
        for rank in range(2):
            for prefix in ("optim", "extra_state"):
                torch.save({"step": 5, "state": torch.arange(16)}, self.actor / f"{prefix}_world_size_2_rank_{rank}.pt")
        torch.save({"position": 32}, self.ckpt / "data.pt")
        (self.ckpt.parent / "latest_checkpointed_iteration.txt").write_text("5")
        self.original = {p: p.read_bytes() for p in self.ckpt.rglob("*") if p.is_file()}
        self.count = 0
        self.store = self.root / "sdpo/shared_checkpoint_bases/v1"
        self.patcher = patch.object(convert.guarded, "QUIET_SECONDS", 0)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def run_command(self, *extra):
        self.count += 1
        args = ["--root", str(self.root), "--report-dir", str(self.root / f"report{self.count}"),
                "--job", self.job.job_id, *extra]
        with contextlib.redirect_stdout(io.StringIO()):
            return convert.main(args)

    def test_dry_run_never_changes_checkpoints_or_creates_baseline(self):
        self.assertEqual(self.run_command(), 0)
        self.assertFalse(self.store.exists())
        for path, raw in self.original.items():
            self.assertEqual(path.read_bytes(), raw)

    def test_convert_only_model_and_verify_expand_roundtrip(self):
        self.assertEqual(self.run_command("--apply", "--jobs-paused"), 0)
        self.assertEqual(len(list(self.store.glob("*/base.pt"))), 2)
        for path, raw in self.original.items():
            if path in self.models:
                self.assertTrue(convert.codec.is_shared(path))
                with convert.codec.SharedReader(path) as reader:
                    self.assertEqual(reader.read(), raw)
            else:
                self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(self.run_command("--mode", "verify"), 0)
        self.assertEqual(self.run_command("--apply", "--jobs-paused", "--mode", "expand"), 0)
        for path, raw in self.original.items():
            self.assertEqual(path.read_bytes(), raw)

    def test_apply_requires_stopped_acknowledgement(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_command("--apply")

    def test_missing_rank_refuses_before_rewriting_anything(self):
        self.models[-1].unlink()
        self.assertEqual(self.run_command("--apply", "--jobs-paused"), 1)
        self.assertFalse(self.store.exists())
        self.assertEqual(self.models[0].read_bytes(), self.original[self.models[0]])

    def test_busy_pipeline_prevents_conversion(self):
        import fcntl
        (self.run / "state").mkdir()
        with (self.run / "state/pipeline.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.assertEqual(self.run_command("--apply", "--jobs-paused"), 1)
        self.assertFalse(self.store.exists())

    def test_recent_files_prevent_conversion(self):
        with patch.object(convert.guarded, "QUIET_SECONDS", 300):
            self.assertEqual(self.run_command("--apply", "--jobs-paused"), 1)
        self.assertFalse(self.store.exists())

    def test_symlink_model_refused(self):
        target = self.models[0]
        target.unlink()
        target.symlink_to(self.models[1])
        self.assertEqual(self.run_command("--apply", "--jobs-paused"), 1)
        self.assertFalse(self.store.exists())

    def test_ambiguous_or_unknown_job_refused(self):
        (self.parent / "another-rho0.7-refw0.7-sync0").mkdir()
        with self.assertRaises(ValueError):
            list(convert.current_runs(self.root, {"unknown"}))
        with self.assertRaises(convert.UnsafeCleanup):
            list(convert.current_runs(self.root, {self.job.job_id}))

    def test_partial_conversion_can_be_retried(self):
        original_compact = convert.codec.compact_model
        calls = 0

        def interrupted(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("interrupted")
            return original_compact(*args)

        with patch.object(convert.codec, "compact_model", side_effect=interrupted):
            self.assertEqual(self.run_command("--apply", "--jobs-paused"), 1)
        self.assertTrue(convert.codec.is_shared(self.models[0]))
        self.assertFalse(convert.codec.is_shared(self.models[1]))
        self.assertEqual(self.run_command("--apply", "--jobs-paused"), 0)
        for path in self.models:
            with convert.codec.SharedReader(path) as reader:
                self.assertEqual(reader.read(), self.original[path])

    def test_other_projects_and_trainer_adapters_untouched(self):
        other = self.root / "mcrl/model.pt"
        other.parent.mkdir()
        other.write_bytes(b"unrelated")
        job, parent = next((j, p) for j, p in convert.audit.current_jobs(self.root) if j.job_id == "math_grpo_4b")
        path = parent / "grpo-4b-fixture/checkpoints/grpo-4b-fixture/checkpoint-5/adapter_model.safetensors"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"adapter")
        self.assertEqual(self.run_command("--apply", "--jobs-paused", "--job", job.job_id), 0)
        self.assertEqual(other.read_bytes(), b"unrelated")
        self.assertEqual(path.read_bytes(), b"adapter")

    def test_cleanup_readiness_rejects_missing_shared_dependency(self):
        cleanup = importlib.import_module("cleanup_opsd_extreme")
        for name in ("tokenizer_config.json", "tokenizer.json"):
            (self.actor / "huggingface" / name).write_text("{}")
        self.assertEqual(self.run_command("--apply", "--jobs-paused"), 0)
        self.assertTrue(cleanup.native_ready(self.root, self.ckpt))
        next(self.store.glob("*/base.pt")).unlink()
        self.assertFalse(cleanup.native_ready(self.root, self.ckpt))


if __name__ == "__main__":
    unittest.main()
