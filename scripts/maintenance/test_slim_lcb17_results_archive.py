import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import slim_lcb17_results_archive as slim


RUN = "logs/lcbv6-sr_opsd-Qwen3-1.7B-seed0-alpha0.25-rho0.95"


class SlimLcbArchiveTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.archive = self.root / slim.ARCHIVE_NAME
        self.report = self.root / "receipts"
        # Random content ensures dropping the dumps genuinely shrinks the fixture.
        self.entries = [
            (f"./{RUN}/metrics.jsonl", b'{"step": 5, "acc": 0.7}\n'),
            (f"./{RUN}/train.log", b"original log\n"),
            (f"./{RUN}/launcher.log", b"launcher log\n"),
            ("./config/resolved.yaml", b"seed: 0\n"),
            (f"./{RUN}/rollouts/1.jsonl", os.urandom(10000)),
            (f"./{RUN}/validation/5.jsonl", os.urandom(10000)),
        ]
        self.make_archive()

    def make_archive(self, entries=None, extra=None):
        with tarfile.open(self.archive, "w:gz") as target:
            for name, content in self.entries if entries is None else entries:
                info = tarfile.TarInfo(name)
                info.size = len(content)
                info.mode, info.uid, info.gid, info.mtime = 0o640, 23, 42, 12345
                target.addfile(info, io.BytesIO(content))
            if extra is not None:
                target.addfile(extra)

    def execute(self, apply=True):
        with patch("builtins.print"):
            return slim.slim_archive(self.archive, self.report, apply=apply)

    def assert_original_after_error(self, error=Exception):
        before = self.archive.read_bytes()
        with self.assertRaises(error):
            self.execute()
        self.assertEqual(before, self.archive.read_bytes())
        self.assertFalse(list(self.root.glob("*.part")))

    def test_apply_replaces_archive_preserving_exact_metrics_logs_configs(self):
        before = self.archive.read_bytes()
        original_size = len(before)
        # Existing digest must not keep claiming to describe the old archive.
        checksum = Path(str(self.archive) + ".sha256")
        checksum.write_text(hashlib.sha256(before).hexdigest())
        result = self.execute()
        self.assertEqual(result["status"], "REPLACED_ORIGINAL")
        self.assertEqual(len(result["dropped"]), 2)
        self.assertEqual(len(result["kept"]), 4)
        self.assertLess(self.archive.stat().st_size, original_size)
        with tarfile.open(self.archive) as target:
            for name, content in self.entries[:4]:
                info = target.getmember(name)
                self.assertEqual(target.extractfile(info).read(), content)
                self.assertEqual((info.mode, info.uid, info.gid, info.mtime), (0o640, 23, 42, 12345))
        self.assertEqual(json.loads((self.report / "complete.json").read_text()), result)
        self.assertEqual(checksum.read_text().split()[0], hashlib.sha256(self.archive.read_bytes()).hexdigest())
        self.assertFalse(list(self.root.glob("*.part")))
        self.assertFalse(any(p.read_bytes() == before for p in self.root.iterdir() if p.is_file()))

    def test_audit_never_changes_archive(self):
        before = self.archive.read_bytes()
        self.assertEqual(self.execute(False)["status"], "AUDIT")
        self.assertEqual(before, self.archive.read_bytes())
        self.assertFalse(list(self.root.glob("*.part")))

    def test_gnu_long_names_and_multi_chunk_members(self):
        owner = RUN + "-sync0-entropy1e-6-ema0.05-topk100-steps420-trainbs32-mbs32-rolloutn8-lr1e-5-utilguard-v2"
        metrics = b'{"step": 5, "acc": 0.7}\n' * 100000
        with tarfile.open(self.archive, "w:gz", format=tarfile.GNU_FORMAT) as target:
            for name, content in ((f"./{owner}/metrics.jsonl", metrics),
                                  (f"./{owner}/rollouts/1.jsonl", os.urandom(slim.CHUNK * 3))):
                info = tarfile.TarInfo(name)
                info.size = len(content)
                target.addfile(info, io.BytesIO(content))
        result = self.execute()
        self.assertEqual(result["kept"][f"{owner}/metrics.jsonl"]["sha256"],
                         hashlib.sha256(metrics).hexdigest())
        with tarfile.open(self.archive) as archive:
            self.assertEqual(archive.extractfile(f"./{owner}/metrics.jsonl").read(), metrics)

    def test_rerun_does_not_destroy_slim_archive(self):
        self.execute()
        before = self.archive.read_bytes()
        self.report = self.root / "second_receipt"
        self.assertEqual(self.execute()["status"], "NO_CHANGE")
        self.assertEqual(before, self.archive.read_bytes())

    def test_other_project_and_similarly_named_members_are_kept(self):
        self.entries.extend([
            ("logs/current-math/rollouts/1.jsonl", b"current responses"),
            (f"{RUN}/validation_config.json", b"config"),
            (f"{RUN}/other/validation/1.jsonl", b"unclassified"),
            ("physics/raw_logits_audit/step0.npz", b"logits"),
            ("current/checkpoints/actor.pt", b"resume"),
        ])
        self.make_archive()
        result = self.execute()
        self.assertEqual(len(result["dropped"]), 2)
        for name, _ in self.entries[6:]:
            self.assertIn(name, result["kept"])

    def test_empty_dump_directories_are_removed(self):
        directory = tarfile.TarInfo(f"{RUN}/rollouts/")
        directory.type = tarfile.DIRTYPE
        self.make_archive(extra=directory)
        self.assertIn(f"{RUN}/rollouts", self.execute()["dropped"])

    def test_corrupt_gzip_footer_even_after_tar_end_never_replaces(self):
        data = bytearray(self.archive.read_bytes())
        data[-8] ^= 0xFF
        self.archive.write_bytes(data)
        self.assert_original_after_error(gzip.BadGzipFile)

    def test_missing_gzip_footer_never_replaces(self):
        self.archive.write_bytes(self.archive.read_bytes()[:-8])
        self.assert_original_after_error(EOFError)

    def test_truncated_payload_never_replaces(self):
        data = self.archive.read_bytes()
        self.archive.write_bytes(data[:len(data) // 2])
        self.assert_original_after_error()

    def test_trailing_non_padding_or_concatenated_tar_is_rejected(self):
        raw = gzip.decompress(self.archive.read_bytes())
        self.archive.write_bytes(gzip.compress(raw + b"hidden payload"))
        self.assert_original_after_error(slim.UnsafeArchive)

    def test_duplicate_normalized_path_is_rejected(self):
        self.make_archive(self.entries + [(f"{RUN}/metrics.jsonl", b"other")])
        self.assert_original_after_error(slim.UnsafeArchive)

    def test_unsafe_paths_are_rejected_without_extraction(self):
        for name in ("../outside", "/absolute", "logs/../evidence", "logs\\hidden"):
            with self.subTest(name=name):
                self.make_archive(self.entries + [(name, b"danger")])
                self.assert_original_after_error(slim.UnsafeArchive)
        self.assertFalse((self.root.parent / "outside").exists())

    def test_links_and_special_files_are_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE):
            with self.subTest(kind=kind):
                member = tarfile.TarInfo("pointer")
                member.type, member.linkname = kind, "missing"
                self.make_archive(extra=member)
                self.assert_original_after_error(slim.UnsafeArchive)

    def test_source_symlink_is_not_followed(self):
        real = self.root / "real.tar.gz"
        self.archive.rename(real)
        self.archive.symlink_to(real)
        self.assert_original_after_error(OSError)
        self.assertTrue(self.archive.is_symlink())

    def test_missing_metrics_global_or_per_run_is_rejected(self):
        self.make_archive(self.entries[1:])
        self.assert_original_after_error(slim.UnsafeArchive)
        self.make_archive(self.entries + [
            ("logs/lcbv6-sr_opsd-Qwen3-1.7B-other/validation/5.jsonl", b"responses")])
        self.assert_original_after_error(slim.UnsafeArchive)

    def test_verification_failure_keeps_original(self):
        with patch.object(slim, "verify_replacement", side_effect=slim.UnsafeArchive("bad verification")):
            self.assert_original_after_error(slim.UnsafeArchive)

    def test_interruption_before_replace_keeps_original(self):
        with patch.object(slim, "verify_replacement", side_effect=KeyboardInterrupt()):
            self.assert_original_after_error(KeyboardInterrupt)

    def test_member_corruption_detected_by_verification(self):
        with self.archive.open("rb") as source, patch("builtins.print"):
            expected = slim.scan_archive(source)["kept"]
        self.make_archive([(n, c + b"changed") for n, c in self.entries[:4]])
        with self.assertRaises(slim.UnsafeArchive), patch("builtins.print"):
            slim.verify_replacement(self.archive, expected)

    def test_source_mutation_before_replace_is_detected(self):
        original_verify = slim.verify_replacement
        before = self.archive.read_bytes()

        def mutate(path, expected):
            result = original_verify(path, expected)
            with self.archive.open("ab") as stream:
                stream.write(b"new data")
            return result

        with patch.object(slim, "verify_replacement", side_effect=mutate):
            with self.assertRaises(slim.UnsafeArchive):
                self.execute()
        self.assertEqual(self.archive.read_bytes(), before + b"new data")
        self.assertFalse((self.report / "complete.json").exists())

    def test_lock_prevents_duplicate_writer(self):
        with Path(str(self.archive) + ".slim.lock").open("wb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assert_original_after_error(BlockingIOError)

    def test_write_failure_preserves_original_and_cleans_own_part(self):
        with patch.object(tarfile.TarFile, "addfile", side_effect=OSError("No space left on device")):
            self.assert_original_after_error(OSError)

    def test_receipt_failure_before_replace_preserves_original(self):
        with patch.object(slim, "write_receipt", side_effect=OSError("No space left on device")):
            self.assert_original_after_error(OSError)

    def test_requires_exact_reviewed_basename(self):
        other = self.root / "current_math.tar.gz"
        self.archive.rename(other)
        self.archive = other
        self.assert_original_after_error(slim.UnsafeArchive)

    def test_cli_apply(self):
        result = subprocess.run([sys.executable, slim.__file__, "--archive", str(self.archive),
                                 "--report-dir", str(self.report), "--apply"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Old archive replaced", result.stdout)


if __name__ == "__main__":
    unittest.main()
