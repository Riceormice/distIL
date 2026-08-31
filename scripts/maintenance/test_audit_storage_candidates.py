import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from audit_storage_candidates import classify, discover, disk_bytes


class StorageAuditTests(unittest.TestCase):
    def test_protected_and_review_classifications(self):
        for name in ("models", "envs", "datasets", ".cache"):
            self.assertTrue(classify(name, True, 0, 1).startswith("KEEP_"))
        self.assertEqual(classify("model_states", True, 0, 1), "REVIEW_WEIGHT_CONTAINER")
        self.assertEqual(classify("raw_logits_audit", True, 0, 1), "KEEP_RAW_LOGITS")
        self.assertIsNone(classify("failed_retry", True, 0, 1))
        self.assertEqual(classify("export.tar.gz.part", False, 123, 1), "REVIEW_PARTIAL_ARCHIVE")

    def test_discovery_prunes_evidence_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in ("models/base/checkpoints", "exp/evaluations/checkpoint-5", "exp/raw_logits_audit/step_0000"):
                (root / path).mkdir(parents=True)
                (root / path / "model.pt").write_bytes(b"protected")
            weights = root / "exp/run/model_states/global_step_420"
            weights.mkdir(parents=True)
            marker = root / "exp/run/state/run.complete"
            marker.parent.mkdir()
            marker.touch()
            rows, errors = discover(root, 8, 1, root / "reports-out")
            self.assertFalse(errors)
            review = [row for row in rows if row["kind"].startswith("REVIEW_")]
            self.assertEqual(len(review), 1)
            self.assertEqual(review[0]["path"], str(weights.parent))
            self.assertEqual(review[0]["marker_hints"], [str(marker)])
            self.assertEqual(review[0]["activity"], "UNKNOWN_ON_OTHER_MACHINES")

    def test_nested_sdpo_method_is_not_mistaken_for_shared_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sdpo/envs").mkdir(parents=True)
            weights = root / "math_exp/sdpo/run/native/checkpoints"
            weights.mkdir(parents=True)
            rows, errors = discover(root, 8, 1, root / "report")
            self.assertFalse(errors)
            review = [row for row in rows if row["kind"].startswith("REVIEW_")]
            self.assertEqual([row["path"] for row in review], [str(weights)])
            shared = next(row for row in rows if row["path"] == str(root / "sdpo"))
            self.assertEqual(shared["kind"], "KEEP_RUNTIME_MODEL_DATA")

    def test_no_completion_inference_or_symlink_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "failed_retry/checkpoints"
            weights.mkdir(parents=True)
            (root / "alias").symlink_to(weights, target_is_directory=True)
            rows, errors = discover(root, 8, 1, root / "report")
            self.assertFalse(errors)
            review = next(row for row in rows if row["kind"].startswith("REVIEW_"))
            self.assertEqual(review["marker_hints"], [])
            self.assertEqual(sum(row["kind"] == "KEEP_SYMLINK_NOT_FOLLOWED" for row in rows), 1)

    def test_depth_limits_remain_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "project/run/unknown/deep").mkdir(parents=True)
            rows, _ = discover(root, 2, 1, root / "report")
            self.assertEqual(rows[0]["kind"], "REVIEW_DEPTH_LIMIT")
            self.assertEqual(rows[0]["path"], str(root / "project/run"))

    def test_report_is_excluded_and_size_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "audit-out"
            report.mkdir()
            (report / "old.tar.gz").write_bytes(b"audit")
            archive = root / "incomplete.tar.gz.part"
            archive.write_bytes(b"payload")
            rows, _ = discover(root, 8, 1, report)
            self.assertEqual(len(rows), 1)
            size, error = disk_bytes(archive, 10)
            self.assertIsNotNone(size)
            self.assertFalse(error)
            self.assertEqual(archive.read_bytes(), b"payload")

    def test_cli_reports_without_deleting(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "input"
            weights = root / "exp/checkpoints/model.pt"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"keep this")
            result = subprocess.run([
                sys.executable, str(Path(__file__).with_name("audit_storage_candidates.py")),
                str(root), "--report-dir", str(base / "reports"),
            ], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(weights.read_bytes(), b"keep this")
            report = next((base / "reports").glob("*.json"))
            payload = json.loads(report.read_text())
            self.assertEqual(payload["rows"][0]["kind"], "REVIEW_WEIGHT_CONTAINER")
            self.assertFalse(payload["errors"])


if __name__ == "__main__":
    unittest.main()
