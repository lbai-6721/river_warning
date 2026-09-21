"""Portable paths, immutable run directories and experiment provenance."""
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def portable(path):
    # Lexical normalization avoids opening every source file through Windows
    # GetFinalPathNameByHandle just to write an inventory. No symlink mutation.
    p = Path(os.path.abspath(resolve(path)))
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def read_csv(path):
    with resolve(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        raise ValueError("Cannot write a table without columns")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if hasattr(obj, "item"):
        return clean_json(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path, obj):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(obj), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(resolve(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with resolve(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def table_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()


def new_run(path, config, inputs=()):
    path = resolve(path)
    path.mkdir(parents=True, exist_ok=False)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                              cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = "unavailable", None
    versions = {}
    from importlib.metadata import version, PackageNotFoundError
    for name in ["torch", "torchvision", "numpy", "scipy", "scikit-learn", "Pillow"]:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    fingerprints = {portable(p): digest(p) for p in inputs}
    for p in inputs:
        if resolve(p).suffix.lower() == ".csv":
            mappings = {r.get("mask_mapping") for r in read_csv(p)} - {None, ""}
            fingerprints.update({portable(m): digest(m) for m in mappings})
    write_json(path / "run.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "git_commit": sha, "git_dirty": dirty,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": versions, "inputs": fingerprints,
        "status": "started",
    })
    return path


def seed_all(seed, threads=2):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(name):
    import torch
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for verification")
    return torch.device(name)


def load_weights(path):
    import torch
    # Only tensors and primitive metadata are used by riverlab checkpoints.
    return torch.load(resolve(path), map_location="cpu", weights_only=True)


def completed(path):
    meta = read_json(path / "run.json")
    meta["status"] = "complete"
    meta["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path / "run.json", meta)
