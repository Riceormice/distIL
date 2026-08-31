"""Lossless shared-baseline storage for Torch model shards (stdlib-only codec).

Identical tensor blocks reference one immutable baseline per model/world/rank.
Changed blocks remain checkpoint-local, optionally as compressed XOR deltas.
This preserves EMA rounding, dtype, FSDP metadata and tensor storage aliases;
it does not assume that a teacher's non-LoRA weights are frozen bit-for-bit.
Optimizer/RNG/data files are intentionally outside this codec.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import tempfile
import zipfile
import zlib


MAGIC = b"SDPO-SHARED-V1\n"
BLOCK = 1024 * 1024
MAX_MANIFEST = 32 * 1024 * 1024
ARCH_KEYS = ("model_type", "architectures", "vocab_size", "hidden_size", "intermediate_size",
             "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim")


def namespace(config: dict, world: int, rank: int) -> str:
    identity = {key: config.get(key) for key in ARCH_KEYS}
    identity.update(world=world, rank=rank)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stamp(path: Path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"not a regular file: {path}")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def stream_hash(source) -> str:
    source.seek(0)
    result = hashlib.sha256()
    while chunk := source.read(BLOCK):
        result.update(chunk)
    return result.hexdigest()


def fsync_dir(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def segments(path: Path):
    """Split at Torch ZIP tensor boundaries, unaffected by pickle-header size."""
    size = path.stat().st_size
    regions = []
    with zipfile.ZipFile(path) as archive, path.open("rb") as source:
        for info in archive.infolist():
            name = info.filename.split("/", 1)[-1]
            if not name.startswith("data/") or info.file_size < 65536:
                continue
            if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
                raise ValueError("only unencrypted, uncompressed Torch tensor ZIP entries are supported")
            source.seek(info.header_offset)
            header = source.read(30)
            if len(header) != 30 or header[:4] != b"PK\x03\x04":
                raise ValueError("invalid ZIP local header")
            name_len, extra_len = struct.unpack_from("<HH", header, 26)
            start = info.header_offset + 30 + name_len + extra_len
            regions.append((start, start + info.file_size, name))
    cursor = 0
    for start, end, name in sorted(regions):
        if not cursor <= start <= end <= size:
            raise ValueError("overlapping or out-of-range ZIP tensor entries")
        while cursor < start:
            length = min(BLOCK, start - cursor)
            yield cursor, length, None
            cursor += length
        part = 0
        while cursor < end:
            length = min(BLOCK, end - cursor)
            yield cursor, length, f"{name}:{part}"
            cursor += length
            part += 1
    while cursor < size:
        length = min(BLOCK, size - cursor)
        yield cursor, length, None
        cursor += length


def baseline(source: Path, store: Path, group: str):
    if len(group) != 64 or any(c not in "0123456789abcdef" for c in group):
        raise ValueError("invalid baseline namespace")
    store = store.absolute()
    folder = store / group
    folder.mkdir(parents=True, exist_ok=True)
    base = folder / "base.pt"
    index = folder / "index.json"
    if folder.is_symlink() or base.is_symlink() or index.is_symlink():
        raise ValueError("baseline paths must not be symlinks")
    with (folder / "build.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not base.exists():
            if index.exists():
                raise ValueError(f"baseline missing; refusing to replace an existing identity: {base}")
            before = stamp(source)
            fd, name = tempfile.mkstemp(prefix="base.", suffix=".part", dir=folder)
            temporary = Path(name)
            try:
                with source.open("rb") as inp, os.fdopen(fd, "wb") as out:
                    shutil.copyfileobj(inp, out, BLOCK)
                    out.flush()
                    os.fsync(out.fileno())
                if stamp(source) != before:
                    raise ValueError("source changed while creating shared baseline")
                os.chmod(temporary, 0o444)
                os.replace(temporary, base)
                fsync_dir(folder)
            finally:
                temporary.unlink(missing_ok=True)
        if not index.exists():
            entries = {}
            with base.open("rb") as inp:
                for offset, length, key in segments(base):
                    if key is not None:
                        inp.seek(offset)
                        entries[key] = [offset, length, digest(inp.read(length))]
                sha = stream_hash(inp)
            payload = {"schema": 1, "size": base.stat().st_size, "sha256": sha, "entries": entries}
            fd, name = tempfile.mkstemp(prefix="index.", suffix=".part", dir=folder)
            temporary = Path(name)
            try:
                with os.fdopen(fd, "w") as out:
                    json.dump(payload, out, sort_keys=True)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temporary, index)
                fsync_dir(folder)
            finally:
                temporary.unlink(missing_ok=True)
        payload = json.loads(index.read_text())
        if payload.get("schema") != 1 or payload.get("size") != stamp(base)[2]:
            raise ValueError("baseline index size/schema mismatch")
    return base, payload


def xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("XOR block size mismatch")
    return (int.from_bytes(left, "little") ^ int.from_bytes(right, "little")).to_bytes(len(left), "little")


class SharedReader(io.RawIOBase):
    """Seekable reconstruction accepted by torch.load; never materializes a model."""

    def __init__(self, path):
        super().__init__()
        self.source = self.base = None
        self.cache = OrderedDict()
        try:
            self.source = open(path, "rb")
            if self.source.read(len(MAGIC)) != MAGIC:
                raise ValueError("not a shared checkpoint")
            raw = self.source.read(8)
            if len(raw) != 8:
                raise ValueError("truncated shared header")
            length = struct.unpack("<Q", raw)[0]
            if not 0 < length <= MAX_MANIFEST:
                raise ValueError("invalid shared manifest size")
            self.manifest = json.loads(self.source.read(length))
            self.payload_offset = self.source.tell()
            if self.manifest.get("schema") != 1:
                raise ValueError("unsupported shared checkpoint schema")
            self.base_path = Path(self.manifest["base_path"])
            base_size = stamp(self.base_path)[2]
            self.base = self.base_path.open("rb")
            source_size = os.fstat(self.source.fileno()).st_size
            self.records = self.manifest["segments"]
            self.starts = []
            total = 0
            for record in self.records:
                n, kind, offset, stored, sha, base_offset, base_sha = record
                if type(n) is not int or not 0 < n <= BLOCK or kind not in ("base", "local", "xor"):
                    raise ValueError("invalid shared segment")
                if not isinstance(sha, str) or len(sha) != 64:
                    raise ValueError("missing segment hash")
                if kind != "base" and (type(offset) is not int or type(stored) is not int or
                                       not 0 <= offset <= offset + stored <= source_size - self.payload_offset):
                    raise ValueError("private segment outside file")
                if kind == "local" and stored != n:
                    raise ValueError("private segment size mismatch")
                if kind != "local" and (type(base_offset) is not int or not 0 <= base_offset <= base_size - n):
                    raise ValueError("baseline segment outside file")
                self.starts.append(total)
                total += n
            if type(self.manifest["logical_size"]) is not int or total != self.manifest["logical_size"]:
                raise ValueError("reconstructed size mismatch")
            self.size = total
            self.position = 0
        except BaseException:
            self.close()
            raise

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self._checkClosed()
        if whence not in (0, 1, 2):
            raise ValueError("invalid seek origin")
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def _block(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        n, kind, offset, stored, sha, base_offset, base_sha = self.records[i]
        if kind != "local":
            self.base.seek(base_offset)
            original = self.base.read(n)
            if len(original) != n or digest(original) != base_sha:
                raise ValueError(f"shared baseline corruption: {self.base_path} offset={base_offset}")
        if kind == "base":
            data = original
        else:
            self.source.seek(self.payload_offset + offset)
            data = self.source.read(stored)
            if len(data) != stored:
                raise ValueError("truncated private segment")
            if kind == "xor":
                decoder = zlib.decompressobj()
                delta = decoder.decompress(data, n + 1)
                if len(delta) != n or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                    raise ValueError("invalid XOR delta")
                data = xor_bytes(original, delta)
        if digest(data) != sha:
            raise ValueError("reconstructed segment checksum mismatch")
        self.cache[i] = data
        if len(self.cache) > 8:
            self.cache.popitem(last=False)
        return data

    def readinto(self, buffer):
        self._checkClosed()
        view = memoryview(buffer).cast("B")
        length = min(len(view), max(0, self.size - self.position))
        copied = 0
        while copied < length:
            i = bisect.bisect_right(self.starts, self.position) - 1
            block = self._block(i)
            offset = self.position - self.starts[i]
            take = min(length - copied, len(block) - offset)
            view[copied:copied + take] = block[offset:offset + take]
            copied += take
            self.position += take
        return copied

    def read(self, size=-1):
        size = max(0, self.size - self.position) if size is None or size < 0 else min(size, max(0, self.size - self.position))
        data = bytearray(size)
        self.readinto(data)
        return bytes(data)

    def close(self):
        for stream in (self.source, self.base):
            if stream is not None:
                stream.close()
        self.cache.clear()
        super().close()


def is_shared(path) -> bool:
    with open(path, "rb") as stream:
        return stream.read(len(MAGIC)) == MAGIC


@contextmanager
def checkpoint_stream(path):
    with (SharedReader(path) if is_shared(path) else open(path, "rb")) as source:
        yield source


def load_model(path, **kwargs):
    import torch

    if not is_shared(path):
        return torch.load(path, **kwargs)
    if kwargs.get("mmap"):
        raise ValueError("shared checkpoints require streaming load, not mmap")
    with SharedReader(path) as source:
        return torch.load(source, **kwargs)


def verify_model(path):
    """Hash the complete reconstructed archive without importing Torch."""
    before = stamp(Path(path))
    with checkpoint_stream(path) as source:
        sha = stream_hash(source)
        if isinstance(source, SharedReader) and sha != source.manifest["logical_sha256"]:
            raise ValueError("shared checkpoint failed full verification")
        size = source.seek(0, 2)
        base = str(source.base_path) if isinstance(source, SharedReader) else None
    if stamp(Path(path)) != before:
        raise ValueError("checkpoint changed during verification")
    return {"logical_sha256": sha, "logical_bytes": size, "base_path": base}


def expand_model(path):
    """Atomically restore a standalone, byte-identical ordinary Torch archive."""
    path = Path(path)
    before = stamp(path)
    if not is_shared(path):
        return {"status": "already_plain", "before_bytes": before[2], "after_bytes": before[2]}
    fd, name = tempfile.mkstemp(prefix=path.name + ".expand.", suffix=".part", dir=path.parent)
    temporary = Path(name)
    try:
        with SharedReader(path) as source, os.fdopen(fd, "wb") as out:
            os.fchmod(out.fileno(), stat.S_IMODE(path.stat().st_mode))
            shutil.copyfileobj(source, out, BLOCK)
            expected = source.manifest["logical_sha256"]
            out.flush()
            os.fsync(out.fileno())
        with temporary.open("rb") as check:
            if stream_hash(check) != expected:
                raise ValueError("expanded archive failed verification")
        if stamp(path) != before:
            raise ValueError("source changed; original not replaced")
        size = temporary.stat().st_size
        os.replace(temporary, path)
        fsync_dir(path.parent)
        return {"status": "expanded", "before_bytes": before[2], "after_bytes": size,
                "logical_sha256": expected}
    finally:
        temporary.unlink(missing_ok=True)


def compact_model(path, store, group):
    """Verify reconstructed original bytes before atomically replacing this shard."""
    path = Path(path)
    before = stamp(path)
    if is_shared(path):
        with SharedReader(path) as source:
            if stream_hash(source) != source.manifest["logical_sha256"]:
                raise ValueError("existing shared checkpoint failed verification")
        return {"status": "already_shared", "before_bytes": before[2], "after_bytes": before[2]}
    layout = list(segments(path))
    if not any(key is not None for _, _, key in layout):
        return {"status": "no_large_tensors", "before_bytes": before[2], "after_bytes": before[2]}
    base_path, index = baseline(path, Path(store), group)
    by_hash = {(length, sha): offset for offset, length, sha in index["entries"].values()}
    private = result = None
    try:
        fd, name = tempfile.mkstemp(prefix=path.name + ".private.", suffix=".part", dir=path.parent)
        private = Path(name)
        records = []
        sha = hashlib.sha256()
        shared_bytes = delta_bytes = 0
        with path.open("rb") as inp, base_path.open("rb") as base, os.fdopen(fd, "wb") as out:
            for offset, length, key in layout:
                inp.seek(offset)
                data = inp.read(length)
                if len(data) != length:
                    raise ValueError("source truncated during compaction")
                sha.update(data)
                checksum = digest(data)
                base_offset = by_hash.get((length, checksum)) if key is not None else None
                if base_offset is not None:
                    records.append([length, "base", 0, 0, checksum, base_offset, checksum])
                    shared_bytes += length
                    continue
                payload, kind, base_sha = data, "local", None
                candidate = index["entries"].get(key)
                if candidate and candidate[1] == length:
                    base_offset, _, base_sha = candidate
                    base.seek(base_offset)
                    base_data = base.read(length)
                    if digest(base_data) != base_sha:
                        raise ValueError("baseline data changed")
                    delta = zlib.compress(xor_bytes(data, base_data), level=1)
                    if len(delta) < length * 0.85:
                        payload, kind = delta, "xor"
                        delta_bytes += length
                records.append([length, kind, out.tell(), len(payload), checksum, base_offset, base_sha])
                out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        manifest = {"schema": 1, "logical_size": before[2], "logical_sha256": sha.hexdigest(),
                    "base_path": str(base_path), "base_sha256": index["sha256"], "segments": records}
        header = json.dumps(manifest, separators=(",", ":")).encode()
        if len(header) > MAX_MANIFEST:
            raise ValueError("shared manifest exceeds size limit")
        final_size = len(MAGIC) + 8 + len(header) + private.stat().st_size
        if final_size >= before[2]:
            return {"status": "not_smaller", "before_bytes": before[2], "after_bytes": before[2]}
        fd, name = tempfile.mkstemp(prefix=path.name + ".shared.", suffix=".part", dir=path.parent)
        result = Path(name)
        with os.fdopen(fd, "wb") as out, private.open("rb") as inp:
            os.fchmod(out.fileno(), stat.S_IMODE(path.stat().st_mode))
            out.write(MAGIC + struct.pack("<Q", len(header)) + header)
            shutil.copyfileobj(inp, out, BLOCK)
            out.flush()
            os.fsync(out.fileno())
        with SharedReader(result) as reconstructed:
            if stream_hash(reconstructed) != manifest["logical_sha256"]:
                raise ValueError("lossless reconstruction check failed")
        if stamp(path) != before:
            raise ValueError("source changed; original not replaced")
        os.replace(result, path)
        result = None
        fsync_dir(path.parent)
        return {"status": "shared", "before_bytes": before[2], "after_bytes": final_size,
                "base_path": str(base_path), "logical_sha256": manifest["logical_sha256"],
                "shared_bytes": shared_bytes, "delta_bytes": delta_bytes}
    finally:
        for temporary in (private, result):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def compact_saved_model(path, config: dict, world: int, rank: int):
    store = os.environ.get("SDPO_SHARED_CHECKPOINT_STORE")
    if not store:
        return
    result = compact_model(path, store, namespace(config, world, rank))
    print(f"SHARED_CHECKPOINT {path}: {json.dumps(result, sort_keys=True)}", flush=True)
