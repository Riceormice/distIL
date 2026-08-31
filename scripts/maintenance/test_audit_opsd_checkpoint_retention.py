#!/usr/bin/env python3
"""Safety/retention tests for the read-only OPSD storage audit."""

import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import audit_opsd_checkpoint_retention as audit


CURRENT = "sr_opsd_math_alpha_rho_sweep_legacy_allprompts_20260825"
RUN_NAME = "sr-opsd-8b-rho0.7-refw0.9-test"


class RetentionAuditTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.home = Path(directory.name)
        self.root = self.home / "data"
        self.root.mkdir()

    def run_dir(self, experiment=CURRENT):
        path = self.root / experiment / "sr_opsd" / RUN_NAME
        (path / "state").mkdir(parents=True, exist_ok=True)
        return path

    def checkpoint(self, run, step):
        path = run / "native/checkpoints" / run.name / f"global_step_{step}"
        (path / "actor").mkdir(parents=True)
        (path / "actor/model.pt").write_bytes(b"fixture only")
        (path / "data.pt").write_bytes(b"fixture only")
        return path

    def rows(self, experiment=CURRENT):
        rows, retained, errors = audit.discover(self.root / experiment, audit.scopes(self.root)[experiment])
        self.assertEqual(errors, [])
        audit.classify_rows(self.root, rows)
        return rows, retained

    def test_scope_excludes_other_projects_and_shared_assets(self):
        scopes = audit.scopes(self.root)
        self.assertIn(CURRENT, scopes)
        self.assertIn("sr_opsd_math_4b_alpha_rho_sweep_eval5_n16_a800_20260829", scopes)
        self.assertIn("math_grpo_4b_opsd_trl_aligned_eval5_n16_a800_20260827", scopes)
        for name in ("sdpo", "models", "envs", "code", "mcrl", "result_archives",
                     "online_marq_v31_seed0_dense_llama31_8b_gptq_16gpu_v1_20260824"):
            self.assertNotIn(name, scopes)

    def test_paused_current_latest_is_kept(self):
        run = self.run_dir()
        self.checkpoint(run, 50)
        self.checkpoint(run, 55)
        rows, _ = self.rows()
        policies = {row["step"]: row["policy"] for row in rows}
        self.assertEqual(policies, {50: "REVIEW_CURRENT_OLDER", 55: "KEEP_CURRENT_LATEST"})
        self.assertTrue(all(row["job"] == "math_alpha090_rho070" for row in rows))
        self.assertTrue(all("NOT verified" in row["resume_hint"] for row in rows))

    def test_old_checkpoint_without_valid_new_one_is_not_claimed_deletable(self):
        run = self.run_dir()
        self.checkpoint(run, 50)
        path = self.checkpoint(run, 55)
        (path / "data.pt").unlink()
        rows, _ = self.rows()
        old = next(row for row in rows if row["step"] == 50)
        latest = next(row for row in rows if row["step"] == 55)
        self.assertEqual(old["policy"], "KEEP_NEWER_UNVERIFIED")
        self.assertIn("restore", old["reason"])
        self.assertEqual(latest["resume_hint"], "unknown")
        self.assertEqual(latest["policy"], "KEEP_CURRENT_LATEST")
        self.assertFalse(any(row["policy"].startswith("DELETE") for row in rows))

    def test_busy_checkpoint_is_kept(self):
        run = self.run_dir()
        self.checkpoint(run, 50)
        lock = run / "state/pipeline.lock"
        with lock.open("w") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            rows, _ = self.rows()
            self.assertEqual(rows[0]["policy"], "KEEP_BUSY")

    def test_nightly_lock_also_protects_checkpoints(self):
        run = self.run_dir()
        self.checkpoint(run, 50)
        lock = self.root / "nightly_experiment_state/math_alpha090_rho070/nightly.lock"
        lock.parent.mkdir(parents=True)
        with lock.open("w") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            rows, _ = self.rows()
            self.assertEqual(rows[0]["policy"], "KEEP_BUSY")

    def test_merged_export_waits_for_evaluation(self):
        run = self.run_dir()
        self.checkpoint(run, 50)
        (run / "merged/checkpoint-50").mkdir(parents=True)
        rows, _ = self.rows()
        self.assertEqual(next(row for row in rows if row["kind"] == "merged")["policy"],
                         "KEEP_EVALUATION_PENDING")
        for dataset in audit.DATASETS:
            path = run / "evaluations/checkpoint-50" / f"{dataset}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("not validated")
        rows, _ = self.rows()
        merged = next(row for row in rows if row["kind"] == "merged")
        self.assertEqual(merged["policy"], "REVIEW_EVALUATED_EXPORT")
        self.assertIn("validate", merged["reason"])

    def test_completion_marker_requires_evidence_review(self):
        run = self.run_dir()
        self.checkpoint(run, 100)
        (run / "state/complete").touch()
        rows, _ = self.rows()
        self.assertEqual(rows[0]["policy"], "REVIEW_COMPLETED_RUN")
        self.assertIn("verify all", rows[0]["reason"])

    def test_historical_runs_are_candidates_not_automatically_deleted(self):
        name = audit.HISTORICAL_ROOTS[0]
        run = self.run_dir(name)
        checkpoint = self.checkpoint(run, 60)
        rows, _ = self.rows(name)
        self.assertEqual(rows[0]["policy"], "REVIEW_HISTORICAL_WEIGHTS")
        self.assertIn("custom-output", rows[0]["reason"])
        self.assertTrue(checkpoint.is_dir())

    def test_evidence_and_symlinks_are_not_descended(self):
        run = self.run_dir()
        self.checkpoint(run, 50)
        for name in ("raw_logits_audit", "topk_probe", "generation", "evaluations"):
            (run / name / "checkpoints/global_step_999").mkdir(parents=True)
        outside = self.home / "unrelated"
        outside.mkdir()
        (run / "alias").symlink_to(outside, target_is_directory=True)
        rows, retained = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertTrue(any(row["policy"] == "KEEP_SYMLINK_NOT_FOLLOWED" for row in retained))
        self.assertFalse(any(str(outside) in row["path"] for row in rows))

    def test_unmapped_run_in_current_root_is_protected(self):
        run = self.root / CURRENT / "another-method/unknown-run"
        (run / "state").mkdir(parents=True)
        self.checkpoint(run, 50)
        rows, _ = self.rows()
        self.assertEqual(rows[0]["policy"], "KEEP_UNMAPPED")

    def test_current_trainer_checkpoint_is_retained(self):
        name = "math_4b_opsd_grouped8x8_eval5_n16_a800_20260827"
        run = self.root / name / "opsd/opsd-4b-fixture"
        (run / "state").mkdir(parents=True)
        checkpoint = run / "checkpoints" / run.name / "checkpoint-20"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 20}))
        (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture only")
        rows, _ = self.rows(name)
        self.assertEqual(rows[0]["policy"], "KEEP_CURRENT_LATEST")
        self.assertEqual(rows[0]["job"], "math_opsd8x8_4b")
        self.assertIn("optimizer/RNG restore NOT verified", rows[0]["resume_hint"])

    def test_physics_weights_do_not_select_raw_logits(self):
        name = "physics_p0_sdpo_fkl_jsd_20260827"
        run = self.root / name / "Qwen3-8B/sdpo_fkl/seed0"
        for path in ("state", "model_states/global_step_420", "raw_logits_audit/step_0420"):
            (run / path).mkdir(parents=True)
        rows, retained = self.rows(name)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["policy"], "REVIEW_PHYSICS_WEIGHTS")
        self.assertEqual(rows[0]["path"], str(run / "model_states"))
        self.assertTrue(any(row["path"] == str(run / "raw_logits_audit") for row in retained))

    def test_cli_writes_report_only(self):
        run = self.run_dir()
        checkpoint = self.checkpoint(run, 50)
        report = self.home / "report"
        result = subprocess.run(
            [sys.executable, audit.__file__, "--root", str(self.root),
             "--report-dir", str(report)], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(checkpoint.is_dir())
        payload = json.loads((report / "audit.json").read_text())
        self.assertEqual(payload["checkpoints"][0]["policy"], "KEEP_CURRENT_LATEST")
        self.assertTrue((report / "summary.txt").is_file())
        self.assertEqual(payload["errors"], [])
        self.assertEqual(list((run / "state").iterdir()), [])

    def test_apply_is_not_supported(self):
        result = subprocess.run(
            [sys.executable, audit.__file__, "--root", str(self.root),
             "--report-dir", str(self.home / "report"), "--apply"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.home / "report").exists())


if __name__ == "__main__":
    unittest.main()
