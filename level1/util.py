"""Small helpers: hashing, atomic writes, YAML, git state."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_files(root: Path, files) -> str:
    """One hash for a set of files: their paths relative to root and their contents."""
    h = hashlib.sha256()
    for p in sorted(files, key=lambda q: Path(q).relative_to(root).as_posix()):
        h.update(Path(p).relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(sha256_file(p).encode())
        h.update(b"\0")
    return h.hexdigest()


def atomic_write(path: Path, data: str | bytes) -> None:
    """Write a file so that readers never see a partial version."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb" if isinstance(data, bytes) else "w") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    atomic_write(path, json.dumps(obj, indent=1, sort_keys=True) + "\n")


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def append_jsonl(path: Path, obj) -> None:
    """Logs are append-only: one JSON object per line, flushed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass  # a line cut off by a crash
    return out


def read_yaml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def git_state(root: Path = REPO_ROOT) -> dict:
    """The commit a result was produced from, and whether the tree was dirty."""
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None

    head, status = run("rev-parse", "HEAD"), run("status", "--porcelain")
    return {
        "commit": head.stdout.strip() if head is not None and head.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status is not None and status.returncode == 0 else None,
    }
