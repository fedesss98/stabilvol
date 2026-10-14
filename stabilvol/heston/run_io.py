"""Portable manifests and atomic, content-checked research artifacts."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def source_fingerprint(root):
    root = Path(root)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    files = sorted((root / "stabilvol/heston").glob("*.py")) + [root / "scripts/calibrate_heston_corrected.py"]
    return {"revision": revision, "files": {p.relative_to(root).as_posix(): file_digest(p) for p in files if p.exists()}}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(plain(value), handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Windows indexers and virus scanners can briefly lock the destination.
        # Keep the old checkpoint intact and retry the atomic operation itself.
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if os.name != "nt" or attempt == 5:
                    raise
                time.sleep(0.05 * 2 ** attempt)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def immutable_manifest(path, specification):
    document = {"specification": plain(specification), "fingerprint": digest(specification)}
    path = Path(path)
    if path.exists():
        existing = read_manifest(path)
        if existing != document:
            raise ValueError(f"manifest mismatch: {path}; use a new run directory")
    else:
        atomic_json(path, document)
    return document["fingerprint"]


def read_manifest(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("fingerprint") != digest(doc.get("specification")):
        raise ValueError(f"corrupt manifest: {path}")
    return doc


def verify_artifacts(directory, root=None):
    """Use this before loading a corrected recount or notebook visualization."""
    directory = Path(directory)
    manifest = read_manifest(directory / "manifest.json")
    spec = manifest["specification"]
    for name, checksum in spec.get("artifacts", {}).items():
        if file_digest(directory / name) != checksum:
            raise ValueError(f"stale or modified artifact: {name}")
    if root is not None and spec.get("source") != source_fingerprint(root):
        raise ValueError("source differs from run manifest; create a new run")
    return manifest
