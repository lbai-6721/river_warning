"""Inference adapters; historical feature scores never impersonate image inference."""
import hashlib
import importlib.util
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from ..data import timestamp
from ..features import audit_legacy_features, image_quality, legacy_tables, pair_geometry
from ..io import ROOT, digest, resolve
from ..segmentation import load_model, predict_image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def iso_time(value):
    try:
        return datetime.fromisoformat(value).isoformat()
    except (TypeError, ValueError):
        raise ValueError("需要有效的拍摄时间")


def validate_times(a, b):
    try:
        delta = (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()
    except (ValueError, TypeError):
        raise ValueError("两张图的时间格式或时区不一致")
    if delta <= 0:
        raise ValueError("当前图的拍摄时间必须晚于参考图")
    return delta


class AlertState:
    """Consecutive valid observations; unknown never clears an existing alert."""
    def __init__(self, options):
        self.options = options
        self.active = False
        self.high = self.low = 0
        self.last = None

    def update(self, score, when, valid=True):
        now = datetime.fromisoformat(when)
        if self.last is not None:
            gap = (now - self.last).total_seconds()
            if gap <= 0:
                raise ValueError("监控序列必须严格按时间递增")
            if gap > self.options["max_gap_seconds"]:
                self.high = self.low = 0
        self.last = now
        if not valid or score is None:
            self.high = self.low = 0
            return "interrupted" if self.active else "unknown"
        if score >= self.options["threshold"]:
            self.high += 1
            self.low = 0
            if self.high >= self.options["consecutive"]:
                new = not self.active
                self.active = True
                return "triggered" if new else "active"
            return "active" if self.active else "pending"
        self.high = 0
        if score <= self.options["clear_threshold"]:
            self.low += 1
        else:
            self.low = 0
        if self.active and self.low >= self.options["clear_consecutive"]:
            self.active = False
            return "cleared"
        return "active" if self.active else "idle"


class Catalog:
    def __init__(self, config):
        self.config = config
        self.files, self.pairs, self.legacy = {}, [], {}
        dataset = resolve(config["dataset"])
        for mode in ("day", "night"):
            root = dataset / mode
            if not root.is_dir():
                continue
            for folder in sorted(root.glob("*/*")):
                if not folder.is_dir():
                    continue
                frames = []
                for path in folder.iterdir():
                    if path.suffix.lower() in IMAGE_SUFFIXES:
                        try:
                            frames.append((timestamp(path.name), path))
                        except ValueError:
                            continue
                frames.sort()
                for (ta, a), (tb, b) in zip(frames, frames[1:]):
                    if ta >= tb:
                        continue
                    aid, bid = self.register(a), self.register(b)
                    pair_id = hashlib.sha256((aid + bid).encode()).hexdigest()[:20]
                    self.pairs.append(dict(id=pair_id, name=folder.name, mode=mode,
                                           camera_id=mode, reference=aid, current=bid,
                                           timestamp_a=ta, timestamp=tb, legacy_available=False))
        self._legacy_index()
        self.pairs.sort(key=lambda p: (p["mode"], p["timestamp"], p["id"]))
        self.lookup = {p["id"]: p for p in self.pairs}

    def register(self, path):
        path = Path(path).resolve()
        key = hashlib.sha256(str(path).encode()).hexdigest()[:24]
        self.files[key] = path
        return key

    def _legacy_index(self):
        folder = resolve(self.config["legacy_features"])
        self.legacy_issue = None
        if not folder.is_dir():
            self.legacy_issue = "历史特征目录不存在"
            return
        try:
            audit = audit_legacy_features(folder)
            up, down, area = legacy_tables(folder)
            entries = {}
            for row in audit["rows"]:
                # The six unresolved records are excluded. Known folder-hour
                # errors may retain their exact frame identity, with provenance.
                if row["status"] not in {"hour_match", "reference_hour_only"}:
                    continue
                i = row["area_csv_row"] - 2
                a, b = 1 + 2*i, 2 + 2*i
                x = np.asarray([up[b][1:], down[b][1:]], dtype=np.float32)
                y = np.asarray(area[i+1][1:11], dtype=np.float32)
                if x.shape != (2, 20) or y.shape != (10,) or not (
                        np.isfinite(x).all() and np.isfinite(y).all()):
                    continue
                key = (up[a][0], up[b][0])
                entries.setdefault(key, []).append((x, y, row["area_csv_row"]))
            for pair in self.pairs:
                key = tuple(self.files[pair[k]].name for k in ("reference", "current"))
                matches = entries.get(key, [])
                if len(matches) == 1:
                    self.legacy[pair["id"]] = matches[0]
                    pair["legacy_available"] = True
            self.legacy_hashes = audit["source_sha256"]
        except (OSError, ValueError, KeyError, IndexError) as exc:
            self.legacy_issue = str(exc)


class Engine:
    def __init__(self, config, catalog):
        self.config, self.catalog = config, catalog
        self.models, self.classifiers = {}, {}
        self.cache = OrderedDict()
        self.legacy_model = None
        torch.set_num_threads(config.get("threads", 2))
        self.device = torch.device(config.get("device", "cpu"))

    def segment(self, file_id, mode):
        profile = self.config["profiles"][mode]
        if mode not in self.models:
            checkpoint = resolve(profile["checkpoint"])
            model, cp = load_model(checkpoint, self.device)
            self.models[mode] = (model, cp, digest(checkpoint))
        model, cp, sha = self.models[mode]
        path = self.catalog.files[file_id]
        stat = path.stat()
        key = (file_id, mode, stat.st_mtime_ns, stat.st_size)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        with Image.open(path) as source:
            im = ImageOps.exif_transpose(source).convert("RGB")
        roi = profile.get("roi")
        if roi:
            roi_size = (roi[2] - roi[0], roi[3] - roi[1])
            if im.size != roi_size:
                if list(im.size) != profile["source_size"]:
                    raise ValueError("白天模型需要 2560×1440 原图或 2560×1024 ROI；请先核对摄像头配置")
                im = im.crop(tuple(roi))
        probability = predict_image(model, im, cp["config"], self.device)
        mask = (probability >= .5).astype(np.uint8)
        result = (im, mask, sha)
        self.cache[key] = result
        while len(self.cache) > 6:
            self.cache.popitem(last=False)
        return result

    def legacy_score(self, pair_id):
        if self.legacy_model is None:
            spec = importlib.util.spec_from_file_location(
                "river_monitor_legacy", ROOT / "Classifier_ACC0.93/cnn/net.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cp = resolve(self.config["legacy_checkpoint"])
            model = module.CNN().eval()
            model.load_state_dict(torch.load(cp, map_location="cpu", weights_only=True))
            self.legacy_sha = digest(cp)
            self.legacy_model = model
        x, y, row = self.catalog.legacy[pair_id]
        with torch.inference_mode():
            score = self.legacy_model(torch.from_numpy(x[None]), torch.from_numpy(y[None])).softmax(1)[0, 1]
        return float(score), self.legacy_sha, row

    def image_classifier(self, mode, masks, images, segmentation_sha):
        from ..classification import Predictor, accepted
        import joblib
        if mode not in self.classifiers:
            cp = resolve(self.config["profiles"][mode]["classifier"])
            payload = joblib.load(cp)  # Only operator-configured local trusted files.
            meta = payload["upstream"]
            if (meta.get("source") != "predicted" or meta.get("feature_kind") != "pair-features"
                    or meta.get("checkpoint_sha256") != segmentation_sha):
                raise ValueError("分类器与分割权重或两帧特征版本不匹配")
            self.classifiers[mode] = (payload, Predictor(payload, self.device), digest(cp))
        payload, predictor, sha = self.classifiers[mode]
        meta = payload["upstream"]
        features = pair_geometry(masks, images, meta.get("bins", 20), meta.get("sigma", 0),
                                 meta.get("quality_config", {}))
        if not accepted([features], payload["config"])[0]:
            return None, sha, float(payload["threshold"])
        return float(predictor([features])[0]), sha, float(payload["threshold"])

    def analyze(self, pair, output):
        elapsed = validate_times(pair["timestamp_a"], pair["timestamp"])
        mode = pair["mode"]
        images, masks = [], []
        for key in ("reference", "current"):
            im, mask, sha = self.segment(pair[key], mode)
            images.append(im)
            masks.append(mask)
        if masks[0].shape != masks[1].shape:
            raise ValueError("前后图像的分析区域尺寸不一致")
        quality = [image_quality(im, **self.config["quality"]) for im in images]
        reasons = []
        if not all(q["quality_valid"] for q in quality):
            reasons.append("图像过暗、过亮或模糊")
        if any(not m.any() for m in masks):
            reasons.append("未提取到有效河体，需检查图像与模型")
        features = pair_geometry(masks, images, bins=20)
        area = [float(m.mean()) for m in masks]
        score, cls_sha, source, source_row = None, None, "unavailable", None
        threshold = self.config["alert"]["threshold"]
        profile = self.config["profiles"][mode]
        if not reasons:
            if profile.get("classifier"):
                score, cls_sha, threshold = self.image_classifier(mode, masks, images, sha)
                source = "image_pipeline"
                if score is None:
                    reasons.append("分类特征未通过模型质量检查")
            elif pair.get("id") in self.catalog.legacy:
                score, cls_sha, source_row = self.legacy_score(pair["id"])
                source = "historical_features"
            else:
                reasons.append("当前图像对没有配套分类模型；分割和变化分析已完成")
        if score is not None and not np.isfinite(score):
            raise ValueError("分类模型返回了无效分数")
        output.mkdir(parents=True, exist_ok=False)
        for name, im, mask in zip(("reference", "current"), images, masks):
            rgb = np.asarray(im).copy()
            overlay = rgb.copy()
            overlay[mask == 1] = (rgb[mask == 1] * .55 + np.array([23, 198, 168]) * .45).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (83, 242, 203), 2)
            preview = Image.fromarray(overlay)
            preview.thumbnail((1280, 800))
            preview.save(output / (name + ".jpg"), quality=88)
            Image.fromarray(mask * 255).save(output / (name + "_mask.png"))
        change = np.asarray(images[1]).copy()
        removed, added = (masks[0] == 1) & (masks[1] == 0), (masks[1] == 1) & (masks[0] == 0)
        change[removed] = (246, 154, 87)
        change[added] = (72, 155, 240)
        preview = Image.fromarray(change)
        preview.thumbnail((1280, 800))
        preview.save(output / "change.jpg", quality=88)
        return dict(pair, score=score, threshold=threshold,
                    prediction=None if score is None else int(score >= threshold),
                    source=source, source_row=source_row, reasons=reasons, quality=quality,
                    area_reference=area[0], area_current=area[1],
                    area_change_pp=(area[1]-area[0])*100,
                    added_fraction=float(added.mean()), removed_fraction=float(removed.mean()),
                    interval_seconds=elapsed, features=features, segmentation_sha256=sha,
                    classifier_sha256=cls_sha,
                    legacy_feature_sha256=getattr(self.catalog, "legacy_hashes", {}) if source == "historical_features" else None)
