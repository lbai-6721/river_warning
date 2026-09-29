"""Train/validation separation and full-image evaluation for all baselines."""
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from . import augment
from .data import ensure_unseen, provenance, validate_splits
from .io import (completed, device_for, digest, load_weights, new_run, portable,
                 read_csv, read_json, resolve, seed_all, table_digest, write_csv, write_json)
from .metrics import boundary_counts, boundary_f1, confusion_seg, segmentation
from .models import segmentation_model
from .masks import read_mask


def tensor_image(image):
    # Explicit [0,1] input normalization for all scratch-trained baselines.
    return torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 255


class SegDataset(Dataset):
    def __init__(self, rows, config, training=False, seed=42):
        self.rows, self.config, self.training = rows, config, training
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        mask = read_mask(row)
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + index)
        size = int(self.config.get("input_size", 512))
        with Image.open(resolve(row["image_path"])) as im:
            if (im.height, im.width) != mask.shape:
                raise ValueError("Image/mask size mismatch: {}".format(row["sample_id"]))
            if self.training and self.config.get("train_crop", False):
                h, w = mask.shape
                if min(h, w) < size:
                    image = np.array(im.convert("RGB"))
                    image = np.pad(image, ((0, max(0, size-h)), (0, max(0, size-w)), (0, 0)))
                    mask = np.pad(mask, ((0, max(0, size-h)), (0, max(0, size-w))),
                                  constant_values=255)
                    h, w = mask.shape
                    top, left = rng.integers(0, h-size+1), rng.integers(0, w-size+1)
                    image, mask = image[top:top+size, left:left+size], mask[top:top+size, left:left+size]
                else:
                    top, left = rng.integers(0, h-size+1), rng.integers(0, w-size+1)
                    image = np.array(im.crop((left, top, left+size, top+size)).convert("RGB"))
                    mask = mask[top:top+size, left:left+size]
            else:
                image = np.array(im.convert("RGB").resize((size, size), Image.BILINEAR))
                mask = np.array(Image.fromarray(mask).resize((size, size), Image.NEAREST))
        if self.training:
            image, mask = augment.apply(image, mask, rng, self.config.get("augmentation", {}))
        return tensor_image(image), torch.from_numpy(mask.astype(np.int64))


def segmentation_loss(logits, target, config):
    valid = target != 255
    if not valid.any():
        return logits.sum() * 0
    ce = F.cross_entropy(logits, target, ignore_index=255, reduction="none")
    gamma = float(config.get("focal_gamma", 0))
    if gamma:
        ce = (1 - torch.exp(-ce)) ** gamma * ce
    loss = ce[valid].mean()
    probability = logits.softmax(1)[:, 1] * valid
    truth = (target == 1).float()
    if config.get("dice_weight", 0):
        dice = 1 - (2*(probability*truth).sum()+1)/(probability.sum()+truth.sum()+1)
        loss = loss + config["dice_weight"] * dice
    if config.get("boundary_weight", 0):
        # Differentiable finite-difference edge matching on valid adjacent pixels.
        terms = []
        for axis in [1, 2]:
            pa, pb = probability.diff(dim=axis), truth.diff(dim=axis)
            adjacent = valid.narrow(axis, 1, valid.shape[axis]-1) & valid.narrow(axis, 0, valid.shape[axis]-1)
            if adjacent.any():
                terms.append((pa-pb).abs()[adjacent].mean())
        if terms:
            loss = loss + config["boundary_weight"] * sum(terms) / len(terms)
    return loss


def train(config, output, device="cuda"):
    if int(config.get("epochs", 100)) < 1:
        raise ValueError("epochs must be positive")
    patience = int(config.get("patience", 20))
    if patience < 0:
        raise ValueError("patience must be nonnegative; use 0 to disable early stopping")
    rows = read_csv(config["manifest"])
    validate_splits(rows)
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    seed_all(int(config.get("seed", 42)), int(config.get("threads", 2)))
    dev = device_for(device)
    run = new_run(output, config, [config["manifest"]])
    model = segmentation_model(config["model"]).to(dev)
    if config.get("initial_weights"):
        initial = load_weights(config["initial_weights"])
        # Explicit strict loading prevents silently missing backbone weights.
        model.load_state_dict(initial.get("state_dict", initial), strict=True)
        write_json(run / "initial_weights.json", {"sha256": digest(config["initial_weights"]),
                   "note": "External initialization provenance must be reviewed before publication"})
    dataset = SegDataset(tr, config, True, config.get("seed", 42))
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    batch_size = int(config.get("batch_size", 4))
    if batch_size < 2 or len(tr) < 2:
        raise ValueError("Training needs >=2 samples/batch for BatchNorm baselines")
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=config.get("workers", 0), generator=generator,
                              drop_last=len(tr) % batch_size == 1)
    val_loader = DataLoader(SegDataset(va, config), batch_size=batch_size, shuffle=False,
                            num_workers=config.get("workers", 0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", .0003),
                                  weight_decay=config.get("weight_decay", .0001))
    history, best, stale = [], float("inf"), 0
    for epoch in range(int(config.get("epochs", 100))):
        dataset.epoch = epoch
        model.train()
        total_loss, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(x), y, config.get("loss", {}))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite segmentation loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(x)
            count += len(x)
        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                val_loss += float(segmentation_loss(model(x), y, config.get("loss", {}))) * len(x)
                val_count += len(x)
        val_loss /= val_count
        history.append({"epoch": epoch+1, "train_loss": total_loss/count, "val_loss": val_loss})
        if val_loss < best:
            best, stale = val_loss, 0
            checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "config": config, "epoch": epoch+1, "val_loss": val_loss,
                          "provenance": provenance(rows), "manifest_sha256": digest(config["manifest"])}
            torch.save(checkpoint, run / "best.pt")
        else:
            stale += 1
        write_csv(run / "history.csv", history)
        print("epoch={} train_loss={:.5f} val_loss={:.5f}".format(epoch+1, total_loss/count, val_loss), flush=True)
        if patience and stale >= patience:
            break
    completed(run)
    return run


def starts(length, tile, stride):
    if not 0 < stride <= tile:
        raise ValueError("Require 0 < stride <= tile_size")
    return list(dict.fromkeys(list(range(0, max(1, length-tile+1), stride)) + [max(0, length-tile)]))


def predict_image(model, image, config, device):
    """Full-image probability map; tile contributions are averaged before argmax."""
    image = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    h, w = image.shape[:2]
    size = int(config.get("input_size", 512))
    tile = int(config.get("tile_size", 0))
    with torch.inference_mode():
        if not tile:
            small = Image.fromarray(image).resize((size, size), Image.BILINEAR)
            logits = model(tensor_image(small).unsqueeze(0).to(device))
            probability = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).softmax(1)
            return probability[0, 1].cpu().numpy()
        stride = int(config.get("tile_stride", tile))
        accum, weights = np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)
        for y in starts(h, tile, stride):
            for x in starts(w, tile, stride):
                patch = image[y:y+tile, x:x+tile]
                ph, pw = patch.shape[:2]
                padded = np.pad(patch, ((0, tile-ph), (0, tile-pw), (0, 0)))
                small = Image.fromarray(padded).resize((size, size), Image.BILINEAR)
                logits = model(tensor_image(small).unsqueeze(0).to(device))
                probability = F.interpolate(logits, size=(tile, tile), mode="bilinear",
                                             align_corners=False).softmax(1)[0, 1].cpu().numpy()
                accum[y:y+ph, x:x+pw] += probability[:ph, :pw]
                weights[y:y+ph, x:x+pw] += 1
        if (weights == 0).any():
            raise RuntimeError("Uncovered pixels during tiled inference")
        return accum / weights


def load_model(checkpoint, device):
    cp = load_weights(checkpoint)
    if "config" not in cp or "provenance" not in cp:
        raise ValueError("Expected riverlab checkpoint with config and provenance")
    model = segmentation_model(cp["config"]["model"]).to(device)
    model.load_state_dict(cp["state_dict"], strict=True)
    model.eval()
    return model, cp


def evaluate(checkpoint, manifest, output, device="cpu", split="test", limit=0):
    rows = read_csv(manifest)
    validate_splits(rows)
    rows = [r for r in rows if r["split"] == split]
    if not rows or limit < 0:
        raise ValueError("Nonempty evaluation split and nonnegative limit required")
    if limit:
        rows = rows[:limit]
    dev = device_for(device)
    model, cp = load_model(checkpoint, dev)
    if split == "test":
        ensure_unseen(rows, cp["provenance"])
    elif split == "val":
        ensure_unseen(rows, cp["provenance"], include_validation=False)
    config = cp["config"]
    run = new_run(output, {"checkpoint": portable(resolve(checkpoint)), "split": split,
                           "limit": limit, "inference": config}, [manifest, checkpoint])
    (run / "masks").mkdir()
    hist, boundaries = np.zeros((2, 2), dtype=np.int64), np.zeros(4, dtype=np.int64)
    predictions, timings = [], []
    per_image = []
    cohort_hist, cohort_boundaries, cohort_counts, cohort_groups = {}, {}, {}, {}
    group_hist = {}
    for i, row in enumerate(rows):
        with Image.open(resolve(row["image_path"])) as im:
            image = im.convert("RGB")
        if dev.type == "cuda":
            torch.cuda.synchronize()
        before = time.perf_counter()
        probability = predict_image(model, image, config, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - before
        timings.append(seconds)
        pred = (probability >= .5).astype(np.uint8)
        path = run / "masks" / ("{:07d}.png".format(i))
        Image.fromarray(pred).save(path)
        truth = read_mask(row)
        h = confusion_seg(truth, pred)
        bc = boundary_counts(truth, pred, config.get("boundary_tolerance", 2))
        hist += h
        boundaries += bc
        group_hist.setdefault(row["group_id"], np.zeros((2, 2), dtype=np.int64))
        group_hist[row["group_id"]] += h
        row["month"] = row["timestamp"][:7]
        for field in ["mode", "weather", "month"]:
            key = field+":"+row.get(field, "unknown")
            cohort_hist.setdefault(key, np.zeros((2, 2), dtype=np.int64))
            cohort_boundaries.setdefault(key, np.zeros(4, dtype=np.int64))
            cohort_hist[key] += h
            cohort_boundaries[key] += bc
            cohort_counts[key] = cohort_counts.get(key, 0)+1
            cohort_groups.setdefault(key, set()).add(row["group_id"])
        predictions.append(dict(row, prediction_path=portable(path)))
        per_image.append(dict(sample_id=row["sample_id"], group_id=row["group_id"],
                              mode=row["mode"], weather=row.get("weather", "unknown"),
                              timestamp=row["timestamp"], seconds=seconds,
                              **segmentation(h), boundary_f1=boundary_f1(bc)))
    metrics = segmentation(hist)
    metrics.update({"boundary_f1": boundary_f1(boundaries), "n_images": len(rows),
                    "seconds_median": np.median(timings), "seconds_p95": np.quantile(timings, .95),
                    "limited_smoke_only": bool(limit), "boundary_tolerance_pixels": config.get("boundary_tolerance", 2)})
    if len(group_hist) >= 2:
        rng = np.random.default_rng(42)
        histograms = list(group_hist.values())
        bootstrap = []
        for _ in range(int(config.get("bootstrap", 1000))):
            indices = rng.integers(0, len(histograms), len(histograms))
            value = segmentation(sum(histograms[i] for i in indices))["river_iou"]
            if np.isfinite(value):
                bootstrap.append(value)
        metrics["river_iou_ci95"] = np.quantile(bootstrap, [.025, .975]).tolist() if bootstrap else None
    else:
        metrics["river_iou_ci95"] = None
    metrics["independent_groups"] = len(group_hist)
    write_json(run / "cohorts.json", {
        key: dict(segmentation(value), boundary_f1=boundary_f1(cohort_boundaries[key]),
                  n_images=cohort_counts[key], independent_groups=len(cohort_groups[key]))
        for key, value in cohort_hist.items()})
    write_json(run / "metrics.json", metrics)
    write_csv(run / "predictions.csv", predictions)
    write_csv(run / "per_image.csv", per_image)
    write_json(run / "predictions.meta.json", {
        "source": "predicted", "checkpoint_sha256": digest(checkpoint),
        "provenance": cp["provenance"], "manifest_sha256": digest(manifest),
        "inference": config})
    completed(run)
    return metrics


def benchmark(checkpoint, image_path, output, device="cpu", warmup=3, repeats=10):
    from .features import geometry, image_quality
    if repeats < 1 or warmup < 0:
        raise ValueError("Invalid benchmark repetition count")
    dev = device_for(device)
    model, cp = load_model(checkpoint, dev)
    run = new_run(output, {"warmup": warmup, "repeats": repeats, "device": device},
                  [checkpoint, image_path])
    with Image.open(resolve(image_path)) as im:
        image = im.convert("RGB")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(warmup+repeats):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        before = time.perf_counter()
        probability = predict_image(model, image, cp["config"], dev)
        mask = (probability >= .5).astype(np.uint8)
        geometry(mask)
        image_quality(image)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-before
        if i >= warmup:
            times.append(elapsed)
    result = {"median_seconds": np.median(times), "p95_seconds": np.quantile(times, .95),
              "parameters": sum(p.numel() for p in model.parameters()),
              "peak_cuda_bytes": torch.cuda.max_memory_allocated() if dev.type == "cuda" else None,
              "scope": "in-memory image preprocessing + segmentation + mask geometry + quality; excludes file IO and classifier",
              "image_size": image.size, "device": device}
    write_json(run / "metrics.json", result)
    completed(run)
    return result
