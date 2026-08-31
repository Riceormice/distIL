import copy
import fcntl
import hashlib
import importlib.util
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

import audit_opsd_checkpoint_retention as audit
import cleanup_opsd_extreme as cleanup
import compact_math_evaluations as compact


def evaluation(dataset="aime24", n=16):
    problems = compact.PROBLEMS[dataset]
    gen = {"full_generation": "reasoning " * 100, "correct": True, "formatted": True,
           "predicted_answer": "42", "generated_tokens": 120, "finish_reason": "stop"}
    rows = [{"problem_id": i, "problem": "question", "ground_truth": "42", "val_n": n,
             "full_generation": gen["full_generation"], "generations": [dict(gen) for _ in range(n)],
             "num_correct": n, "majority_vote_correct": True}
            for i in range(problems)]
    return {"dataset": dataset, "num_problems": problems, "val_n": n,
            "total_solutions": problems * n, "average_at_n": problems * n,
            "pass_at_n": problems, "majority_vote_at_n": problems, "formatted_count": problems * n,
            "average_at_n_pct": 100.0, "pass_at_n_pct": 100.0,
            "majority_vote_at_n_pct": 100.0, "format_rate": 100.0,
            "eval_config": {"temperature": 1.0, "max_new_tokens": 38912}, "results": rows}


class ExtremeCleanupTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.root = self.base / "storage"
        self.root.mkdir()
        self.report = self.base / "receipt"
        self.job, parent = next((j, p) for j, p in audit.current_jobs(self.root)
                                if j.job_id == "math_alpha070_rho070")
        name = cleanup.guarded.sweep_name("0.7", "0.7")
        self.run = parent / name
        (self.run / "state").mkdir(parents=True)
        clock = patch.object(cleanup.time, "time", return_value=time.time() + 1000)
        clock.start()
        self.addCleanup(clock.stop)

    def put(self, path, content=b"state"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def checkpoint(self, run=None, step=10, tracker=True):
        run = run or self.run
        ckpt = run / "native/checkpoints" / run.name / f"global_step_{step}"
        self.put(ckpt / "data.pt")
        self.put(ckpt / "actor/fsdp_config.json", b'{"world_size": 8}')
        self.put(ckpt / "actor/huggingface/config.json", b'{"model_type":"qwen3"}')
        self.put(ckpt / "actor/huggingface/tokenizer_config.json", b'{}')
        self.put(ckpt / "actor/huggingface/tokenizer.json", b'{}')
        self.put(ckpt / "actor/ema_teacher/fsdp_config.json", b'{"world_size": 8}')
        for rank in range(8):
            self.put(ckpt / f"actor/ema_teacher/model_world_size_8_rank_{rank}.pt")
        for kind in ("model", "optim", "extra_state"):
            for rank in range(8):
                self.put(ckpt / f"actor/{kind}_world_size_8_rank_{rank}.pt")
        if tracker:
            self.put(ckpt.parent / "latest_checkpointed_iteration.txt", str(step).encode())
        return ckpt

    def export(self, run=None, step=10):
        path = (run or self.run) / "merged" / f"checkpoint-{step}"
        self.put(path / "model.safetensors")
        self.put(path / "config.json", b'{}')
        return path

    def evaluations(self, run=None, step=5, all_datasets=True):
        paths = []
        for dataset in audit.DATASETS if all_datasets else ("aime24",):
            path = (run or self.run) / "evaluations" / f"checkpoint-{step}" / f"{dataset}.json"
            self.put(path, json.dumps(evaluation(dataset)).encode())
            paths.append(path)
        return paths

    def execute(self, apply=True, retire=True):
        with patch("builtins.print"):
            return cleanup.execute(self.root, self.report, apply, retire)

    def test_remove_export_keep_current_resume_and_compact_pending_eval(self):
        checkpoint = self.checkpoint()
        export = self.export()
        files = self.evaluations(step=10, all_datasets=False)
        before = {p: p.read_bytes() for p in checkpoint.rglob("*") if p.is_file()}
        result = self.execute()
        self.assertEqual(result["deleted_trees"], 1)
        self.assertFalse(export.exists())
        self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
        self.assertTrue(cleanup.native_ready(self.root, checkpoint))
        payload = json.loads(files[0].read_text())
        self.assertNotIn("full_generation", payload["results"][0]["generations"][0])
        self.assertEqual(payload["average_at_n_pct"], 100.0)
        self.assertEqual(result["compacted_files"], 1)

    def test_export_without_rebuild_source_and_pending_eval_is_kept(self):
        export = self.export()
        self.execute()
        self.assertTrue(export.exists())

    def test_export_with_complete_eval_can_be_removed_without_source(self):
        export = self.export(step=5)
        self.evaluations()
        self.execute()
        self.assertFalse(export.exists())

    def test_old_checkpoint_requires_evaluation_and_newer_valid_resume(self):
        old = self.checkpoint(step=5)
        new = self.checkpoint(step=10)
        self.evaluations(step=5)
        self.execute()
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())

    def test_old_checkpoint_kept_when_its_eval_missing(self):
        old = self.checkpoint(step=5)
        self.checkpoint(step=10)
        self.execute()
        self.assertTrue(old.exists())

    def test_incomplete_new_shards_preserve_fallback(self):
        old = self.checkpoint(step=5)
        new = self.checkpoint(step=10)
        (new / "actor/optim_world_size_8_rank_7.pt").unlink()
        self.evaluations(step=5)
        self.execute()
        self.assertTrue(old.exists())
        self.assertTrue(new.exists())

    def test_stale_resume_pointer_preserves_fallback(self):
        old = self.checkpoint(step=5)
        self.checkpoint(step=10, tracker=False)
        self.evaluations(step=5)
        self.execute()
        self.assertTrue(old.exists())

    def test_missing_teacher_shard_preserves_fallback(self):
        old = self.checkpoint(step=5)
        new = self.checkpoint(step=10)
        (new / "actor/ema_teacher/model_world_size_8_rank_7.pt").unlink()
        self.evaluations(step=5)
        self.execute()
        self.assertTrue(old.exists())
        self.assertFalse(cleanup.native_ready(self.root, new))

    def test_missing_tokenizer_keeps_only_pending_eval_export(self):
        new = self.checkpoint(step=10)
        (new / "actor/huggingface/tokenizer.json").unlink()
        export = self.export(step=10)
        self.execute()
        self.assertTrue(export.exists())

    def test_trainer_requires_all_optimizer_and_rng_shards(self):
        ckpt = self.base / "storage/test-trainer/checkpoint-10"
        self.put(ckpt / "trainer_state.json", b'{"global_step": 10}')
        self.put(ckpt / "adapter_model.safetensors")
        self.put(ckpt / "latest", b'global_step10')
        self.put(ckpt / "global_step10/mp_rank_00_model_states.pt")
        for rank in range(8):
            self.put(ckpt / f"rng_state_{rank}.pth")
            self.put(ckpt / f"global_step10/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")
        self.assertTrue(cleanup.trainer_ready(self.root, ckpt, 10))
        missing = ckpt / "global_step10/bf16_zero_pp_rank_7_mp_rank_00_optim_states.pt"
        missing.unlink()
        self.assertFalse(cleanup.trainer_ready(self.root, ckpt, 10))
        self.put(missing)
        (ckpt / "rng_state_7.pth").unlink()
        self.assertFalse(cleanup.trainer_ready(self.root, ckpt, 10))

    def test_historical_ahead_checkpoint_retired_only_with_current_resume(self):
        old_run = self.root / cleanup.guarded.OLD_SWEEP / "sr_opsd" / self.run.name
        (old_run / "state").mkdir(parents=True)
        old = self.checkpoint(old_run, step=60)
        new = self.checkpoint(step=10)
        self.execute()
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())

    def test_historical_checkpoint_retained_without_current_counterpart(self):
        old_run = self.root / cleanup.guarded.OLD_SWEEP / "sr_opsd" / self.run.name
        (old_run / "state").mkdir(parents=True)
        old = self.checkpoint(old_run, step=60)
        self.execute()
        self.assertTrue(old.exists())

    def test_historical_retirement_needs_flag(self):
        old_run = self.root / cleanup.guarded.OLD_SWEEP / "sr_opsd" / self.run.name
        (old_run / "state").mkdir(parents=True)
        old = self.checkpoint(old_run, step=60)
        self.checkpoint(step=10)
        self.execute(retire=False)
        self.assertTrue(old.exists())

    def test_busy_pipeline_preserves_weights_and_raw_text(self):
        self.checkpoint()
        export = self.export()
        path = self.evaluations(all_datasets=False)[0]
        before = path.read_bytes()
        with (self.run / "state/pipeline.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.execute()
        self.assertEqual(result["failures_or_skips"], 1)
        self.assertTrue(export.exists())
        self.assertEqual(path.read_bytes(), before)

    def test_busy_nightly_wrapper_preserves_run(self):
        self.checkpoint()
        export = self.export()
        path = self.put(self.root / "nightly_experiment_state" / self.job.job_id / "nightly.lock")
        with path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.execute()
        self.assertTrue(export.exists())

    def test_other_projects_assets_and_physics_are_untouched(self):
        paths = [self.put(self.root / name / "checkpoints/global_step_1/model.pt") for name in (
            "mcrl_gradient_transport_qwen3_1_7b_20260823", "marq_full_surface_atomic",
            "models", "sdpo", "envs", *audit.PHYSICS_ROOTS)]
        paths += [self.put(self.root / audit.PHYSICS_ROOTS[1] / "raw_logits_audit/a.npz"),
                  self.put(self.root / audit.PHYSICS_ROOTS[1] / "generation/step_0.jsonl")]
        before = {p: p.read_bytes() for p in paths}
        self.execute()
        self.assertTrue(all(p.read_bytes() == value for p, value in before.items()))

    def test_symlink_in_export_prevents_deletion(self):
        self.checkpoint()
        export = self.export()
        source = self.put(self.base / "outside")
        (export / "linked.pt").symlink_to(source)
        self.execute()
        self.assertTrue(export.exists())
        self.assertEqual(source.read_bytes(), b"state")

    def test_dry_run_never_changes_weights_or_text(self):
        self.checkpoint()
        export = self.export()
        path = self.evaluations(all_datasets=False)[0]
        before = path.read_bytes()
        self.execute(apply=False)
        self.assertTrue(export.exists())
        self.assertEqual(path.read_bytes(), before)

    def test_recent_files_skip_whole_run(self):
        self.checkpoint()
        export = self.export()
        self.evaluations(all_datasets=False)
        with patch.object(cleanup.time, "time", return_value=os.stat(export).st_mtime + 10):
            self.execute()
        self.assertTrue(export.exists())

    def test_completed_run_requires_all_scheduled_evals(self):
        ckpt = self.checkpoint(step=100)
        self.put(self.run / "state/complete", b"")
        self.evaluations(step=100)
        self.execute()
        self.assertTrue(ckpt.exists())

    def test_complete_run_can_drop_final_weights(self):
        ckpt = self.checkpoint(step=100)
        self.put(self.run / "state/complete", b"")
        for step in range(5, 101, 5):
            self.evaluations(step=step)
        self.execute()
        self.assertFalse(ckpt.exists())
        self.assertTrue(cleanup.run_complete(self.root, self.run, self.job))

    def test_compaction_does_not_change_existing_metric_and_sample_fields(self):
        original = evaluation()
        snapshot = copy.deepcopy(original)
        result, count = compact.compact(original, "aime24", "abc")
        self.assertEqual(original, snapshot)
        self.assertEqual(count, 30 * 17)
        for key, value in original.items():
            if key != "results":
                self.assertEqual(result[key], value)
        for before, after in zip(original["results"], result["results"]):
            for key, value in before.items():
                if key not in {"generations", "full_generation"}:
                    self.assertEqual(after[key], value)
            for b, a in zip(before["generations"], after["generations"]):
                self.assertEqual({k: v for k, v in a.items() if not k.startswith("full_generation")},
                                 {k: v for k, v in b.items() if not k.startswith("full_generation")})
        self.assertEqual(compact.compact(result, "aime24", "def"), (result, 0))

    def test_compaction_duplicate_and_invalid_json_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(ValueError):
                compact.load(raw)
        data = evaluation()
        data["results"][0]["generations"].pop()
        with self.assertRaises(ValueError):
            compact.compact(data, "aime24", "abc")

    def test_compaction_write_failure_preserves_file(self):
        path = self.evaluations(all_datasets=False)[0]
        before = path.read_bytes()
        self.report.mkdir()
        with patch.object(cleanup.os, "replace", side_effect=OSError("failed replace")):
            with self.assertRaises(OSError):
                cleanup.compact_file(self.root, path, self.report, io.StringIO(), True)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(list(path.parent.glob("*.part")))

    def test_validator_and_wandb_header_still_accept_compacted_file(self):
        path = self.evaluations(all_datasets=False)[0]
        uploader_path = Path(cleanup.__file__).parents[1] / "math/wandb_math_8runs/upload_math_8runs_to_wandb.py"
        spec = importlib.util.spec_from_file_location("extreme_test_uploader", uploader_path)
        uploader = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = uploader
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(uploader)
        before = uploader.read_eval_header(path, "aime24", False)
        self.execute()
        after = uploader.read_eval_header(path, "aime24", False)
        self.assertEqual({k: v for k, v in before.items() if k != "fingerprint"},
                         {k: v for k, v in after.items() if k != "fingerprint"})

    def test_protocol_comparison_layout_compacts_without_weights(self):
        run = self.root / "sr_opsd_math_refw_sweep30"
        path = self.put(run / "evaluations/oldrun/checkpoint-5/aime24.json",
                        json.dumps(evaluation(n=64)).encode())
        self.execute()
        data = json.loads(path.read_text())
        self.assertEqual(data["val_n"], 64)
        self.assertNotIn("full_generation", data["results"][0])

    def test_unmapped_current_run_is_not_touched(self):
        run = self.run.parent / "unreviewed-experiment"
        (run / "state").mkdir(parents=True)
        export = self.export(run)
        path = self.evaluations(run, all_datasets=False)[0]
        before = path.read_bytes()
        self.execute()
        self.assertTrue(export.exists())
        self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
