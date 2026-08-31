#!/usr/bin/env python3
"""Discard old LCB training/validation originals, retaining other archive members.

--apply verifies a streamed replacement before atomically replacing the original
archive. The removed generation dumps are NOT backed up. No extraction occurs.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import tempfile
import time


ARCHIVE_NAME = "sdpo_lcbv6_sr_opsd_1_7b_h200_results.tar.gz"
DEFAULT_ARCHIVE = Path("/media/vlm-ckp-fileset/ylong/result_archives") / ARCHIVE_NAME
CHUNK = 1024 * 1024


class UnsafeArchive(RuntimeError):
    pass


def member_path(member: tarfile.TarInfo) -> str:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or "\\" in member.name:
        raise UnsafeArchive(f"unsafe archive path: {member.name!r}")
    if not (member.isfile() or member.isdir()) or member.sparse is not None:
        raise UnsafeArchive(f"unsupported link/special/sparse member: {member.name!r}")
    return str(path)


def is_generation_dump(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return (
        len(parts) >= 3
        and parts[0] == "logs"
        and parts[1].startswith("lcbv6-sr_opsd-Qwen3-1.7B-")
        and parts[2] in {"rollouts", "validation"}
    )


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def scan_archive(source, destination=None, *, reject_generations=False) -> dict:
    kept, dropped, seen = {}, {}, set()
    next_progress = time.monotonic()
    with gzip.GzipFile(fileobj=source, mode="rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|", bufsize=CHUNK) as archive:
            for member in archive:
                path = member_path(member)
                if path in seen:
                    raise UnsafeArchive(f"duplicate normalized member: {path}")
                seen.add(path)
                remove = is_generation_dump(path)
                if remove and reject_generations:
                    raise UnsafeArchive(f"generation dump survived replacement: {path}")
                record = {
                    "size": member.size, "type": "file" if member.isfile() else "directory",
                    "mode": member.mode, "uid": member.uid, "gid": member.gid,
                    "mtime": member.mtime,
                }
                if member.isfile():
                    with archive.extractfile(member) as payload:
                        if remove:
                            # Read in bounded chunks, even for multi-GB response dumps.
                            while payload.read(CHUNK):
                                pass
                        else:
                            reader = HashReader(payload)
                            if destination is not None:
                                destination.addfile(member, reader)
                            else:
                                while reader.read(CHUNK):
                                    pass
                            record["sha256"] = reader.digest.hexdigest()
                elif not remove and destination is not None:
                    destination.addfile(member)
                (dropped if remove else kept)[path] = record
                if time.monotonic() >= next_progress:
                    print(f"scanned={len(seen)} kept={len(kept)} removed={len(dropped)} "
                          f"compressed_read_GiB={source.tell() / 2**30:.2f}", flush=True)
                    next_progress = time.monotonic() + 20

            # tar stops at its end marker; drain the gzip stream to verify its CRC
            # and reject non-padding trailing data that would otherwise be lost.
            while True:
                tail = archive.fileobj.read(CHUNK)
                if not tail:
                    break
                if tail.strip(b"\0"):
                    raise UnsafeArchive("non-padding data after tar end marker")
    return {"kept": kept, "dropped": dropped}


def fingerprint(info) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def ensure_unchanged(path: Path, source, before: list[int]) -> None:
    if (fingerprint(path.lstat()) != before
            or fingerprint(os.fstat(source.fileno())) != before):
        raise UnsafeArchive("source archive changed while processing; not replacing it")


def regular_open(path: Path, flags: int, mode=0o600):
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, mode)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise UnsafeArchive(f"not a regular file: {path}")
    return os.fdopen(fd, "rb" if flags == os.O_RDONLY else "r+b")


def verify_replacement(path: Path, expected: dict) -> str:
    with regular_open(path, os.O_RDONLY) as source:
        actual = scan_archive(source, reject_generations=True)
        if actual["kept"] != expected:
            raise UnsafeArchive("replacement members/content do not match retained originals")
        source.seek(0)
        digest = hashlib.sha256()
        while data := source.read(CHUNK):
            digest.update(data)
        return digest.hexdigest()


def write_receipt(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_checksum(path: Path, digest: str) -> None:
    fd, name = tempfile.mkstemp(prefix=path.name + ".sha256.", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as stream:
            stream.write(f"{digest}  {path.name}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, Path(str(path) + ".sha256"))
    finally:
        temp.unlink(missing_ok=True)


def slim_archive(path: Path, report: Path, *, apply=False) -> dict:
    path, report = path.absolute(), report.absolute()
    if path.name != ARCHIVE_NAME:
        raise UnsafeArchive(f"this script only handles {ARCHIVE_NAME}")
    report.mkdir(parents=True, exist_ok=True)
    if any((report / name).exists() for name in ("ready.json", "complete.json", "audit.json")):
        raise UnsafeArchive("use a new report directory for this invocation")
    temp = None
    try:
        with ExitStack() as stack:
            lock = stack.enter_context(regular_open(Path(str(path) + ".slim.lock"),
                                                   os.O_RDWR | os.O_CREAT))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            source = stack.enter_context(regular_open(path, os.O_RDONLY))
            before_info = os.fstat(source.fileno())
            before = fingerprint(before_info)
            print(f"mode={'APPLY' if apply else 'AUDIT'} archive={path}", flush=True)
            print("Discarding old LCB rollouts + validation originals only. "
                  "Metrics/config/logs outside those trees are retained.", flush=True)
            if apply:
                fd, name = tempfile.mkstemp(prefix=path.name + ".slim.", suffix=".part",
                                            dir=path.parent)
                temp = Path(name)
                with os.fdopen(fd, "wb") as output:
                    os.fchmod(output.fileno(), stat.S_IMODE(before_info.st_mode))
                    with gzip.GzipFile(fileobj=output, mode="wb", compresslevel=6, mtime=0) as zipped:
                        with tarfile.open(fileobj=zipped, mode="w|", bufsize=CHUNK) as target:
                            inventory = scan_archive(source, target)
                    output.flush()
                    os.fsync(output.fileno())
            else:
                inventory = scan_archive(source)
            ensure_unchanged(path, source, before)
            metrics = [p for p in inventory["kept"]
                       if p.endswith("/metrics.jsonl") and inventory["kept"][p]["size"] > 0]
            if not metrics:
                raise UnsafeArchive("no nonempty retained metrics.jsonl found")
            for removed in inventory["dropped"]:
                owner = PurePosixPath(*PurePosixPath(removed).parts[:2])
                if str(owner / "metrics.jsonl") not in metrics:
                    raise UnsafeArchive(f"missing retained metrics for {owner}")
            result = {
                "archive": str(path), "source_fingerprint": before,
                "source_compressed_bytes": before_info.st_size,
                "removed_uncompressed_bytes": sum(x["size"] for x in inventory["dropped"].values()),
                "metrics_files": metrics, **inventory,
            }
            if not apply or not inventory["dropped"]:
                result["status"] = "AUDIT" if not apply else "NO_CHANGE"
                write_receipt(report / "audit.json", result)
                print(f"{result['status']}: removed_members={len(inventory['dropped'])}", flush=True)
                return result

            print("Verifying retained members and gzip integrity...", flush=True)
            result["replacement_sha256"] = verify_replacement(temp, inventory["kept"])
            result["replacement_compressed_bytes"] = temp.stat().st_size
            if temp.stat().st_size >= before_info.st_size:
                raise UnsafeArchive("replacement is not smaller; original retained")
            result["compressed_bytes_reduced"] = before_info.st_size - temp.stat().st_size
            result["status"] = "VERIFIED_READY_TO_REPLACE"
            write_receipt(report / "ready.json", result)
            ensure_unchanged(path, source, before)
            # No delete-first window: this atomically removes the old archive name
            # only after a complete, verified replacement exists on the same FS.
            os.replace(temp, path)
            temp = None
            write_checksum(path, result["replacement_sha256"])
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            result["status"] = "REPLACED_ORIGINAL"
            write_receipt(report / "complete.json", result)
            print(f"DONE: removed_members={len(inventory['dropped'])} "
                  f"kept_members={len(inventory['kept'])} "
                  f"archive_GiB={before_info.st_size / 2**30:.2f} -> {path.stat().st_size / 2**30:.2f}\n"
                  f"Old archive replaced; no raw-generation backup retained.\n"
                  f"Retained archive: {path}\nReceipt: {report / 'complete.json'}", flush=True)
            return result
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="irreversibly discard generation dumps after verification")
    args = parser.parse_args()
    try:
        slim_archive(args.archive, args.report_dir, apply=args.apply)
    except (OSError, EOFError, tarfile.TarError, UnsafeArchive) as exc:
        print(f"ERROR: {exc}", flush=True)
        print("Check the receipt: ready.json alone does not prove replacement; "
              "complete.json records success.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
