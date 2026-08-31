"""Test the production worker's EMA checkpoint hooks without starting Ray."""
import ast
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
VERL = ROOT / "SDPO/verl" if (ROOT / "SDPO/verl").is_dir() else ROOT / "verl"


def worker_method(name):
    source = VERL / "workers/fsdp_workers.py"
    module = ast.parse(source.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    scope = {"os": os, "dist": SimpleNamespace(barrier=Mock())}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    return scope[name]


class TeacherResumeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "actor"
        self.path.mkdir()
        self.worker = SimpleNamespace(
            _is_actor=True, _is_rollout=False, _is_lora=False,
            _is_offload_param=False, _is_offload_optimizer=False,
            checkpoint_manager=Mock(), teacher_checkpoint_manager=Mock(),
        )

    def test_restores_separate_teacher(self):
        teacher = self.path / "ema_teacher"
        teacher.mkdir()
        worker_method("load_checkpoint")(self.worker, str(self.path))
        self.worker.teacher_checkpoint_manager.load_checkpoint.assert_called_once_with(
            local_path=str(teacher), hdfs_path=None, del_local_after_load=False,
        )

    def test_missing_teacher_is_not_silently_reset(self):
        with patch.dict(os.environ, {"SDPO_ALLOW_LEGACY_TEACHER_RESET": ""}):
            with self.assertRaisesRegex(FileNotFoundError, "Exact EMA resume"):
                worker_method("load_checkpoint")(self.worker, str(self.path))
        self.worker.teacher_checkpoint_manager.load_checkpoint.assert_not_called()

    def test_legacy_reset_is_explicit(self):
        with patch.dict(os.environ, {"SDPO_ALLOW_LEGACY_TEACHER_RESET": "1"}), patch("builtins.print") as output:
            worker_method("load_checkpoint")(self.worker, str(self.path))
        self.assertIn("LEGACY_EMA_TEACHER_RESET", output.call_args.args[0])
        self.worker.teacher_checkpoint_manager.load_checkpoint.assert_not_called()

    def test_teacher_save_path_and_non_teacher_jobs(self):
        with patch.dict("sys.modules", {"verl.utils.logger": SimpleNamespace(log_with_rank=Mock())}):
            worker_method("save_checkpoint")(self.worker, str(self.path), global_step=5, max_ckpt_to_keep=2)
        self.worker.teacher_checkpoint_manager.save_checkpoint.assert_called_once_with(
            local_path=str(self.path / "ema_teacher"), hdfs_path=None, global_step=5, max_ckpt_to_keep=None,
        )
        del self.worker.teacher_checkpoint_manager
        worker_method("load_checkpoint")(self.worker, str(self.path))
        self.worker.checkpoint_manager.load_checkpoint.assert_called_once()


if __name__ == "__main__":
    unittest.main()
