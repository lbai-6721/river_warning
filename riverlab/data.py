"""Dataset manifests and grouping before patches or temporal windows."""
import csv
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .io import digest, portable, read_csv, resolve, write_csv, write_json


def timestamp(name):
    matches = re.findall(r"(?<!\d)(20\d{12})(?:\d{3})?(?!\d)", str(name))
    if len(matches) != 1:
        raise ValueError("Expected one YYYYMMDDhhmmss timestamp in {!r}".format(name))
    return datetime.strptime(matches[0], "%Y%m%d%H%M%S").isoformat()


def parent_id(name):
    # IDs contain dots in camera IP addresses, even when they have no extension.
    base = Path(name).name
    if Path(base).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        base = Path(base).stem
    return re.sub(r"_\d{2}$", "", base)


def frame_id(name, camera):
    return camera + ":" + parent_id(name)


def scan_dataset(dataset, camera, mode, hash_images=False, check_sizes=False):
    dataset = resolve(dataset)
    images = sorted((dataset / "JPEGImages").glob("*.jpg"))
    if not images:
        raise ValueError("No JPEGImages/*.jpg in {}".format(dataset))
    masks = {p.stem: p for p in (dataset / "SegmentationClass").glob("*.png")}
    rows = []
    for index, path in enumerate(images, 1):
        mask = masks.get(path.stem)
        if mask is None:
            raise ValueError("Missing mask for {}".format(path))
        width, height = "", ""
        if check_sizes:
            with Image.open(path) as im, Image.open(mask) as m:
                if im.size != m.size:
                    raise ValueError("Image/mask size mismatch: {}".format(path))
                width, height = im.size
        t = timestamp(path.stem)
        rows.append({
            "sample_id": camera + ":" + path.stem, "parent_id": frame_id(path, camera),
            "camera_id": camera, "mode": mode, "timestamp": t,
            "image_path": portable(path), "mask_path": portable(mask),
            "width": width, "height": height, "date": t[:10],
            "event_id": "", "weather": "unknown", "label": "",
            "image_sha256": digest(path) if hash_images else "",
        })
        if index % 5000 == 0:
            print("inventory: {}/{} images checked".format(index, len(images)), flush=True)
    return rows


def scan_pairs(root, camera, mode, label):
    """Inventory an explicitly labelled directory; folder names are not ground truth."""
    rows = []
    for directory in sorted(resolve(root).iterdir()):
        if not directory.is_dir():
            continue
        images = sorted([p for p in directory.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
                        key=lambda p: timestamp(p.stem))
        if len(images) < 2:
            continue
        # Generate only consecutive pairs. Four-frame folders do not silently
        # become all pairwise combinations.
        for a, b in zip(images, images[1:]):
            ta, tb = timestamp(a.stem), timestamp(b.stem)
            if tb <= ta:
                raise ValueError("Non-increasing timestamps in {}".format(directory))
            rows.append({
                "sample_id": camera + ":" + directory.name + ":" + b.stem,
                "frame_a": frame_id(a, camera), "frame_b": frame_id(b, camera),
                "image_a": portable(a), "image_b": portable(b),
                "timestamp_a": ta, "timestamp": tb, "camera_id": camera,
                "mode": mode, "label": label, "event_id": "",
                "label_source": "directory_inventory_unverified",
                "weather": "unknown",
            })
    if not rows:
        raise ValueError("No timestamped image pairs found")
    return rows


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        self.parent[self.find(b)] = self.find(a)


def identities(row):
    keys = []
    for col in ["parent_id", "frame_a", "frame_b"]:
        if row.get(col):
            keys.append(("frame", row[col]))
    if row.get("image_sha256"):
        keys.append(("hash", row["image_sha256"]))
    if row.get("event_id"):
        keys.append(("event", row["event_id"]))
    # Group dates globally, including both endpoints and both cameras.
    for col in ["timestamp", "timestamp_a"]:
        if row.get(col):
            keys.append(("date", row[col][:10]))
    return keys


def group_rows(rows):
    uf = UnionFind(len(rows))
    seen = {}
    for i, row in enumerate(rows):
        for key in identities(row):
            if key in seen:
                uf.union(i, seen[key])
            else:
                seen[key] = i
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[uf.find(i)].append(i)
    result = [dict(r) for r in rows]
    for indices in components.values():
        stable = min(result[i]["sample_id"] for i in indices)
        import hashlib
        gid = "g_" + hashlib.sha256(stable.encode()).hexdigest()[:16]
        for i in indices:
            result[i]["group_id"] = gid
    return result


def validate_splits(rows, require_all=True):
    if not rows:
        raise ValueError("Empty manifest")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample_id")
    seen = {}
    for row in rows:
        split = row.get("split", "")
        if split not in {"train", "val", "test"}:
            raise ValueError("Each row needs train/val/test split")
        if not row.get("group_id"):
            raise ValueError("Missing group_id")
        for key in identities(row) + [("group", row["group_id"])]:
            previous = seen.setdefault(key, split)
            if previous != split:
                raise ValueError("Data leakage across {} and {}: {}".format(previous, split, key))
    counts = Counter(r["split"] for r in rows)
    if require_all and set(counts) != {"train", "val", "test"}:
        raise ValueError("Need nonempty train, val and test groups")
    return {
        "samples": dict(counts),
        "groups": {s: len({r["group_id"] for r in rows if r["split"] == s}) for s in counts},
        "labels": {s: dict(Counter(str(r.get("label", "")) for r in rows if r["split"] == s))
                   for s in counts},
    }


def split_rows(rows, train=0.6, val=0.2, seed=42, method="time", folds=0):
    if not (0 < train < 1 and 0 < val < 1 and train + val < 1):
        raise ValueError("Require positive train/val/test fractions")
    rows = group_rows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    ids = sorted(groups, key=lambda g: min(r["timestamp"] for r in groups[g]))
    if len(ids) < 3:
        raise ValueError("Fewer than three independent date/event/shared-frame groups")
    if method == "group":
        np.random.default_rng(seed).shuffle(ids)
    elif method != "time":
        raise ValueError("Unknown split method")
    # Fractions apply to independent groups, not correlated patches.
    n_train = max(1, min(len(ids) - 2, int(len(ids) * train)))
    n_val = max(1, min(len(ids) - n_train - 1, int(len(ids) * val)))
    assignment = {g: ("train" if i < n_train else
                      "val" if i < n_train + n_val else "test") for i, g in enumerate(ids)}
    for r in rows:
        r["split"] = assignment[r["group_id"]]
    validate_splits(rows)
    if method == "time":
        for a, b in [("train", "val"), ("val", "test")]:
            latest = max(r["timestamp"] for r in rows if r["split"] == a)
            earliest = min(r.get("timestamp_a") or r["timestamp"] for r in rows if r["split"] == b)
            if latest >= earliest:
                raise ValueError("Event/shared-frame groups span chronological boundaries; "
                                 "review intervals or use explicitly retrospective group splitting")
    cv = []
    if folds:
        from sklearn.model_selection import GroupKFold
        dev = [r for r in rows if r["split"] != "test"]
        held = [r for r in rows if r["split"] == "test"]
        if len({r["group_id"] for r in dev}) < folds:
            raise ValueError("Not enough development groups for requested folds")
        for tr, va in GroupKFold(folds).split(dev, groups=[r["group_id"] for r in dev]):
            fold = [dict(dev[i], split="train") for i in tr]
            fold += [dict(dev[i], split="val") for i in va] + [dict(r) for r in held]
            validate_splits(fold)
            cv.append(fold)
    return rows, cv


def attach_events(rows, events):
    """Attach reviewed intervals, without inventing onset or labels."""
    for row in rows:
        matches = set()
        for e in events:
            if e.get("reviewed") != "1" or not e.get("event_id"):
                raise ValueError("Events must have event_id and reviewed=1")
            start, end = e["start"], e["end"]
            if end < start:
                raise ValueError("Event end precedes start")
            if e["camera_id"] != row["camera_id"]:
                continue
            a, b = row.get("timestamp_a", row["timestamp"]), row["timestamp"]
            if a <= end and b >= start:
                matches.add(e["event_id"])
        if len(matches) > 1:
            raise ValueError("A window crosses multiple events; split/review the window")
        if matches:
            row["event_id"] = next(iter(matches))
    return rows


def audit_legacy(voc_root):
    report = {}
    for d in sorted(resolve(voc_root).iterdir()):
        base = d / "ImageSets" / "Segmentation"
        if not base.is_dir():
            continue
        versions = [base] + sorted(base.glob("Fold_*"))
        report[d.name] = {}
        for version in versions:
            sets = {}
            for split in ["train", "val", "test"]:
                p = version / (split + ".txt")
                if split == "test" and not p.exists():
                    p = base / "test.txt"
                sets[split] = set(p.read_text().split()) if p.exists() else set()
            comparisons = {}
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
                comparisons[a + "_" + b] = {
                    "exact": len(sets[a] & sets[b]),
                    "parent": len({parent_id(x) for x in sets[a]} &
                                  {parent_id(x) for x in sets[b]}),
                }
            report[d.name][version.name] = {
                "counts": {k: len(v) for k, v in sets.items()}, "overlaps": comparisons}
    return report


def audit_masks(rows, limit=100):
    """Reject non-binary labels instead of silently ignoring unknown classes."""
    from .masks import read_mask
    results = []
    indices = np.linspace(0, len(rows) - 1, min(limit, len(rows)), dtype=int)
    for i in indices:
        r = rows[i]
        with Image.open(resolve(r["mask_path"])) as im, Image.open(resolve(r["image_path"])) as original:
            if im.size != original.size:
                raise ValueError("Image/mask size mismatch: " + r["sample_id"])
            raw_shape = np.asarray(im).shape
        mask = read_mask(r)
        values = np.unique(mask)
        results.append({"sample_id": r["sample_id"], "values": values.tolist(),
                        "source_shape": list(raw_shape), "mask_mapping": r.get("mask_mapping", "")})
        if mask.ndim != 2 or not set(values.tolist()) <= {0, 1, 255}:
            raise ValueError("Invalid binary mask {}: {}".format(r["mask_path"], values))
    return results


def ensure_unseen(rows, provenance, include_validation=True):
    forbidden_keys = set(provenance.get("train_identity_keys", []))
    if include_validation:
        forbidden_keys.update(provenance.get("val_identity_keys", []))
    actual_keys = {str(k) for r in rows for k in identities(r)}
    if forbidden_keys & actual_keys:
        raise ValueError("Evaluation reuses development frames, dates, hashes or events")
    forbidden = set(provenance.get("train_groups", []))
    if include_validation:
        forbidden.update(provenance.get("val_groups", []))
    overlap = forbidden & {r["group_id"] for r in rows}
    if overlap:
        raise ValueError("Evaluation overlaps model development groups: {}".format(sorted(overlap)[:5]))
    forbidden_ids = set(provenance.get("train_ids", []))
    if include_validation:
        forbidden_ids.update(provenance.get("val_ids", []))
    if forbidden_ids & {r["sample_id"] for r in rows}:
        raise ValueError("Evaluation overlaps model development sample IDs")


def provenance(rows):
    return {
        "train_identity_keys": sorted({str(k) for r in rows if r["split"] == "train" for k in identities(r)}),
        "val_identity_keys": sorted({str(k) for r in rows if r["split"] == "val" for k in identities(r)}),
        "train_groups": sorted({r["group_id"] for r in rows if r["split"] == "train"}),
        "val_groups": sorted({r["group_id"] for r in rows if r["split"] == "val"}),
        "train_ids": sorted(r["sample_id"] for r in rows if r["split"] == "train"),
        "val_ids": sorted(r["sample_id"] for r in rows if r["split"] == "val"),
    }
