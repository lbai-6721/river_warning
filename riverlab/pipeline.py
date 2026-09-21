"""Measured full pair pipeline using a frozen segmenter and classifier."""
import time
from datetime import datetime
import joblib
import numpy as np
import torch
from PIL import Image

from .classification import Predictor, accepted, check_upstream
from .data import ensure_unseen, validate_splits
from .features import pair_geometry
from .io import (completed, device_for, digest, new_run, read_csv, resolve,
                 write_csv, write_json)
from .segmentation import load_model, predict_image


def infer_pairs(seg_checkpoint, cls_checkpoint, pairs_path, frames_path, output,
                device="cpu", split="test", warmup=2):
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    pairs, frames = read_csv(pairs_path), read_csv(frames_path)
    validate_splits(pairs)
    validate_splits(frames)
    selected = [r for r in pairs if r["split"] == split]
    lookup = {r["parent_id"]: r for r in frames}
    if len(lookup) != len(frames):
        raise ValueError("Pipeline needs full-frame/ROI images, not patches")
    payload = joblib.load(resolve(cls_checkpoint))  # Trusted local checkpoints only.
    meta = payload["upstream"]
    if meta.get("source") != "predicted" or meta.get("feature_kind") != "pair-features":
        raise ValueError("Online pair pipeline requires a classifier trained on predicted pair-features")
    if meta.get("checkpoint_sha256") != digest(seg_checkpoint):
        raise ValueError("Use the exact segmenter used to train this classifier")
    dev = device_for(device)
    segmenter, cp = load_model(seg_checkpoint, dev)
    ensure_unseen(selected, payload["provenance"], include_validation=split == "test")
    ensure_unseen(selected, cp["provenance"], include_validation=split == "test")
    if split == "test":
        check_upstream(selected, meta)
    predictor = Predictor(payload, dev)
    threshold = payload["threshold"]
    bins, sigma = meta.get("bins", 20), meta.get("sigma", 0)
    quality = meta.get("quality_config", {})
    for pair in selected:
        if datetime.fromisoformat(pair["timestamp"]) <= datetime.fromisoformat(pair["timestamp_a"]):
            raise ValueError("Pair endpoints must be strictly time ordered")
        for name in ["frame_a", "frame_b"]:
            frame = lookup.get(pair[name])
            if frame is None or frame["camera_id"] != pair["camera_id"] or frame["split"] != split:
                raise ValueError("Missing or inconsistent full-frame entry for pair endpoint")

    def one(pair):
        images, masks = [], []
        for name in ["frame_a", "frame_b"]:
            with Image.open(resolve(lookup[pair[name]]["image_path"])) as im:
                image = im.convert("RGB")
            images.append(image)
            masks.append((predict_image(segmenter, image, cp["config"], dev) >= .5).astype(np.uint8))
        features = dict(pair, **pair_geometry(masks, images, bins, sigma, quality))
        score = float(predictor([features])[0])
        valid = bool(accepted([features], payload["config"])[0])
        return dict(features, score=score, threshold=threshold, valid=int(valid),
                    y_pred=int(valid and score >= threshold))

    run = new_run(output, {"split": split, "device": str(device), "warmup": warmup},
                  [seg_checkpoint, cls_checkpoint, pairs_path, frames_path])
    for _ in range(warmup):
        one(selected[0])
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)
    outputs, times = [], []
    for pair in selected:
        if dev.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = one(pair)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        outputs.append(dict(result, seconds=elapsed))
        times.append(elapsed)
    metrics = {"n_pairs": len(outputs), "median_seconds": np.median(times),
               "p95_seconds": np.quantile(times, .95), "coverage": np.mean([r["valid"] for r in outputs]),
               "segmentation_parameters": sum(p.numel() for p in segmenter.parameters()),
               "peak_cuda_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None,
               "scope": "two image decodes + two segmentations + geometry + quality + classifier; "
                        "excludes model loading, final CSV writing, and event replay",
               "cache": "no cross-pair frame cache; filesystem may be warm",
               "device": str(device), "warmup": warmup}
    write_csv(run/"predictions.csv", outputs)
    write_json(run/"metrics.json", metrics)
    completed(run)
    return metrics
