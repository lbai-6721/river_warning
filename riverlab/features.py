"""Normalized geometry, causal temporal features, and legacy CSV migration."""
import csv
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from .data import group_rows, timestamp
from .io import read_csv, resolve
from .masks import read_mask


def geometry(mask, bins=20, sigma=0.0):
    """All bins including the last one; area is a fraction of fixed image strips.

    Missing boundaries are NaN, never zero displacement. Upper/lower coordinates
    are image-y extrema, not hydrological upstream/downstream boundaries.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or not set(np.unique(mask)) <= {0, 1, 255}:
        raise ValueError("Expected binary 2-D mask, with optional 255 void")
    h, w = mask.shape
    if not 1 <= bins <= w or sigma < 0:
        raise ValueError("bins must be between 1 and image width; sigma must be nonnegative")
    foreground = mask == 1
    known = mask != 255
    present = foreground.any(axis=0)
    upper = np.argmax(foreground, axis=0).astype(float) / h
    lower = (h - 1 - np.argmax(foreground[::-1], axis=0)).astype(float) / h
    upper[~present] = np.nan
    lower[~present] = np.nan
    if sigma:
        # Normalized convolution; do not manufacture boundaries in absent columns.
        for edge in (upper, lower):
            valid = np.isfinite(edge)
            den = gaussian_filter1d(valid.astype(float), sigma, mode="nearest")
            smoothed = gaussian_filter1d(np.nan_to_num(edge), sigma, mode="nearest")
            edge[valid] = smoothed[valid] / den[valid]
    result = {}
    for k, columns in enumerate(np.array_split(np.arange(w), bins)):
        a, b = known[:, columns], foreground[:, columns]
        result["area_{:02d}".format(k)] = float(b.sum() / a.sum()) if a.sum() else np.nan
        result["coverage_{:02d}".format(k)] = float(present[columns].mean())
        for name, edge in [("upper", upper), ("lower", lower)]:
            values = edge[columns]
            result[name + "_{:02d}".format(k)] = (
                float(np.nanmean(values)) if np.isfinite(values).any() else np.nan)
        result["width_{:02d}".format(k)] = (
            result["lower_{:02d}".format(k)] - result["upper_{:02d}".format(k)])
    return result


def boundary_motion(previous, current, bins=20, sigma=0.0):
    """Signed and absolute displacement before averaging, avoiding cancellation."""
    previous, current = np.asarray(previous), np.asarray(current)
    if previous.shape != current.shape:
        raise ValueError("Boundary motion requires the same reference image geometry")
    h, w = current.shape
    if not 1 <= bins <= w or sigma < 0:
        raise ValueError("Invalid bins/sigma")
    valid = (previous == 1).any(0) & (current == 1).any(0)
    out = {}
    for name, reverse in [("upper", False), ("lower", True)]:
        a = previous[::-1] if reverse else previous
        b = current[::-1] if reverse else current
        edges = []
        for mask in [a, b]:
            present = (mask == 1).any(0)
            edge = np.argmax(mask == 1, axis=0).astype(float) / h
            edge[~present] = np.nan
            if sigma:
                denominator = gaussian_filter1d(present.astype(float), sigma, mode="nearest")
                smoothed = gaussian_filter1d(np.nan_to_num(edge), sigma, mode="nearest")
                edge[present] = smoothed[present] / denominator[present]
            edges.append(edge)
        delta = edges[1] - edges[0]
        if reverse:
            delta = -delta
        for k, columns in enumerate(np.array_split(np.arange(w), bins)):
            d = delta[columns][valid[columns]]
            for suffix, values in [("", d), ("abs_", np.abs(d))]:
                out[name + "_" + suffix + "{:02d}".format(k)] = (
                    float(np.mean(values)) if len(values) else np.nan)
    return out


def image_quality(image, blur_min=0.0, dark_max=1.0, bright_max=1.0):
    image = np.asarray(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark, bright = float((gray < 8).mean()), float((gray > 247).mean())
    valid = blur >= blur_min and dark <= dark_max and bright <= bright_max
    return {"quality_valid": int(valid), "blur": blur, "dark_fraction": dark,
            "bright_fraction": bright}


def difference(a, b):
    return {key: float(b[key]) - float(a[key])
            for key in b if key.startswith(("area_", "upper_", "lower_", "width_"))}


def temporal_features(frames, lags=(1,), max_gap_hours=3):
    """Construct strictly causal windows within camera, group, and split."""
    groups = defaultdict(list)
    for frame in frames:
        groups[(frame["camera_id"], frame["group_id"], frame["split"])].append(frame)
    out = []
    lags = sorted(set(int(k) for k in lags))
    if not lags or min(lags) < 1:
        raise ValueError("History lags must be positive")
    for values in groups.values():
        values.sort(key=lambda r: r["timestamp"])
        for i, cur in enumerate(values):
            if i < max(lags):
                continue
            window = values[i - max(lags):i + 1]
            gaps = [(datetime.fromisoformat(b["timestamp"]) -
                     datetime.fromisoformat(a["timestamp"])).total_seconds() / 3600
                    for a, b in zip(window, window[1:])]
            if not all(0 < gap <= max_gap_hours for gap in gaps):
                continue
            row = {k: cur[k] for k in ["sample_id", "group_id", "split", "camera_id",
                                       "timestamp", "mode", "label"]}
            row.update({k: cur.get(k, "") for k in ["event_id", "weather", "parent_id"]})
            row["timestamp_a"] = window[0]["timestamp"]
            row["quality_valid"] = int(all(int(r.get("quality_valid", 1)) for r in window))
            row["edge_coverage"] = min(float(r.get("edge_coverage", 1)) for r in window)
            for lag in lags:
                old = values[i - lag]
                dt = (datetime.fromisoformat(cur["timestamp"]) -
                      datetime.fromisoformat(old["timestamp"])).total_seconds() / 3600
                row["dt_lag{}".format(lag)] = dt
                for key, value in difference(old, cur).items():
                    row[key + "_lag{}".format(lag)] = value
            out.append(row)
    return sorted(out, key=lambda r: (r["timestamp"], r["sample_id"]))


def legacy_tables(folder):
    folder = resolve(folder)
    def rows(name):
        for enc in ["utf-8-sig", "gb18030"]:
            try:
                with (folder / name).open(encoding=enc, newline="") as f:
                    return list(csv.reader(f))
            except UnicodeDecodeError:
                continue
        raise ValueError("Cannot decode {}".format(name))
    return rows("up.csv"), rows("down.csv"), rows("area.csv")


def audit_legacy_features(folder):
    """Report all mismatched rows without guessing which source is correct."""
    up, down, area = legacy_tables(folder)
    issues = []
    counts = {"up_rows": len(up)-1, "down_rows": len(down)-1, "area_rows": len(area)-1}
    if len(up) != len(down) or len(up)-1 != 2*(len(area)-1):
        return {"counts": counts, "valid": False, "issues": [{"error": "row_count_mismatch"}]}
    for i, ar in enumerate(area[1:]):
        ai, bi = 1+2*i, 2+2*i
        issue = {"area_csv_row": i+2, "boundary_csv_rows": [ai+1, bi+1],
                 "area_pair": ar[0], "boundary_frames": [up[ai][0], up[bi][0]]}
        problems = []
        try:
            times = [datetime.fromisoformat(timestamp(up[k][0])) for k in [ai, bi]]
            expected = [datetime.strptime(p, "%Y.%m.%d.%H") for p in ar[0].split("_")]
            if expected != [t.replace(minute=0, second=0) for t in times]:
                problems.append("area_boundary_timestamp_mismatch")
            if times[1] <= times[0]:
                problems.append("non_increasing_timestamps")
            if any(up[k][0] != down[k][0] for k in [ai, bi]):
                problems.append("upper_lower_frame_mismatch")
            if len(ar) != 12 or len(up[bi]) != 21 or len(down[bi]) != 21:
                problems.append("feature_dimension_mismatch")
            elif ar[11] not in {"0", "1"}:
                problems.append("invalid_label")
            for value in ar[1:11]+up[bi][1:]+down[bi][1:]:
                if not np.isfinite(float(value)):
                    problems.append("nonfinite_feature")
                    break
        except (ValueError, IndexError) as exc:
            problems.append(str(exc))
        if problems:
            issue["problems"] = problems
            issues.append(issue)
    return {"counts": counts, "valid": not issues, "issue_pairs": len(issues), "issues": issues}


def load_legacy(folder, camera="day"):
    """Validate identifiers before joining 20-bin edges and 10-bin area data."""
    up, down, area = legacy_tables(folder)
    if len(up) != len(down) or (len(up) - 1) != 2 * (len(area) - 1):
        raise ValueError("Legacy CSV row counts do not align")
    output = []
    for i, ar in enumerate(area[1:]):
        a_idx, b_idx = 1 + 2 * i, 2 + 2 * i
        if [r[0] for r in up[a_idx:b_idx+1]] != [r[0] for r in down[a_idx:b_idx+1]]:
            raise ValueError("up/down frame ID mismatch at pair {}".format(i))
        ta, tb = timestamp(up[a_idx][0]), timestamp(up[b_idx][0])
        # Legacy pair identifiers have hour precision; verify against actual frames.
        pair = ar[0].split("_")
        if len(pair) != 2:
            raise ValueError("Expected start_end area pair ID")
        expected = [datetime.strptime(p, "%Y.%m.%d.%H") for p in pair]
        actual = [datetime.fromisoformat(t).replace(minute=0, second=0) for t in (ta, tb)]
        if expected != actual:
            raise ValueError("Area/boundary time-pair mismatch at {}".format(ar[0]))
        if tb <= ta:
            raise ValueError("Legacy pair is not time ordered")
        if len(ar) != 12 or len(up[b_idx]) != 21 or len(down[b_idx]) != 21:
            raise ValueError("Expected 10 area bins and 20 boundary bins")
        if ar[11] not in {"0", "1"}:
            raise ValueError("Unknown legacy label")
        row = {"sample_id": camera + ":" + ar[0], "timestamp_a": ta, "timestamp": tb,
               "camera_id": camera, "mode": "day" if camera == "day" else camera,
               "frame_a": camera + ":" + up[a_idx][0].rsplit(".", 1)[0],
               "frame_b": camera + ":" + up[b_idx][0].rsplit(".", 1)[0],
               "label": ar[11], "event_id": "", "weather": "unknown",
               "quality_valid": 1, "label_source": "legacy_unverified"}
        for prefix, values in [("area", ar[1:11]), ("upper", up[b_idx][1:]),
                               ("lower", down[b_idx][1:])]:
            for j, value in enumerate(values):
                number = float(value)
                if not np.isfinite(number):
                    raise ValueError("Nonfinite legacy feature")
                row["{}_{:02d}".format(prefix, j)] = number
        row["dt_hours"] = (datetime.fromisoformat(tb) - datetime.fromisoformat(ta)).total_seconds()/3600
        output.append(row)
    # Legacy coordinates are not silently converted to a guessed pixel scale.
    return group_rows(output)


def pair_geometry(masks, images, bins=20, sigma=0.0, quality=None):
    if len(masks) != 2 or len(images) != 2 or any(
            np.asarray(m).shape != np.asarray(im).shape[:2] for m, im in zip(masks, images)):
        raise ValueError("Need two aligned images and masks with identical geometry")
    geometries = [geometry(m, bins, sigma) for m in masks]
    quality_rows = [image_quality(im, **(quality or {})) for im in images]
    row = difference(*geometries)
    row.update(boundary_motion(*masks, bins=bins, sigma=sigma))
    row["quality_valid"] = int(all(q["quality_valid"] for q in quality_rows))
    row["edge_coverage"] = min(
        np.mean([g[k] for k in g if k.startswith("coverage_")]) for g in geometries)
    return row


def extract_pair_features(pairs, predictions, bins=20, sigma=0.0,
                          use_ground_truth=False, quality=None):
    """Use a frame manifest containing image_path and mask_path/prediction_path."""
    lookup = {r["parent_id"]: r for r in predictions}
    if len(lookup) != len(predictions):
        raise ValueError("Pair geometry requires one full image per parent, not patch masks")
    out = []
    for pair in pairs:
        if datetime.fromisoformat(pair["timestamp"]) <= datetime.fromisoformat(pair["timestamp_a"]):
            raise ValueError("Pair endpoints must be strictly time ordered")
        a, b = lookup.get(pair["frame_a"]), lookup.get(pair["frame_b"])
        if a is None or b is None:
            raise ValueError("Missing full-frame mask for {}".format(pair["sample_id"]))
        for frame in [a, b]:
            if frame["camera_id"] != pair["camera_id"]:
                raise ValueError("Pair and frame cameras disagree")
            if pair.get("split") != frame.get("split"):
                raise ValueError("Split pair and segmentation frames together before extraction")
        masks, images = [], []
        for r in [a, b]:
            masks.append(read_mask(r, predicted=not use_ground_truth))
            with Image.open(resolve(r["image_path"])) as im:
                images.append(im.convert("RGB"))
        row = dict(pair)
        row.update(pair_geometry(masks, images, bins, sigma, quality))
        out.append(row)
    return out
