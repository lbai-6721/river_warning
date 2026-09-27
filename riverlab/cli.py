"""Run from the repository root with python -m riverlab."""
import argparse
import json
import sys
from pathlib import Path

from .io import read_csv, read_json, resolve, write_csv, write_json, clean_json


def parser():
    p = argparse.ArgumentParser(description="River imagery research experiments")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Show environment; never trains or downloads")
    q = sub.add_parser("inventory")
    q.add_argument("--config", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--hash-images", action="store_true")
    q.add_argument("--check-sizes", action="store_true")
    q = sub.add_parser("pair-inventory")
    q.add_argument("--root", required=True)
    q.add_argument("--camera", required=True)
    q.add_argument("--mode", choices=["day", "night"], required=True)
    q.add_argument("--label", choices=["0", "1"], required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("merge-manifests")
    q.add_argument("--inputs", nargs="+", required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("split")
    q.add_argument("--manifest", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--events")
    q.add_argument("--method", choices=["time", "group"], default="time")
    q.add_argument("--train", type=float, default=.6)
    q.add_argument("--val", type=float, default=.2)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--folds", type=int, default=0)
    q = sub.add_parser("audit-legacy")
    q.add_argument("--voc", default="segment/VOCdevkit")
    q.add_argument("--output", required=True)
    q = sub.add_parser("check-manifest")
    q.add_argument("--manifest", required=True)
    q.add_argument("--mask-samples", type=int, default=100)
    q = sub.add_parser("import-legacy")
    q.add_argument("--folder", default="Classifier_ACC0.93/cnn")
    q.add_argument("--camera", default="day")
    q.add_argument("--output", required=True)
    q = sub.add_parser("audit-legacy-features")
    q.add_argument("--folder", default="Classifier_ACC0.93/cnn")
    q.add_argument("--output", required=True)
    q.add_argument("--source-pairs")
    q.add_argument("--rows-output")
    q = sub.add_parser("joint-split")
    q.add_argument("--frames", required=True)
    q.add_argument("--pairs", required=True)
    q.add_argument("--events")
    q.add_argument("--output", required=True)
    q.add_argument("--method", choices=["time", "group"], default="time")
    q.add_argument("--seed", type=int, default=42)
    q = sub.add_parser("reconstruct")
    q.add_argument("--manifest", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--columns", type=int, default=5)
    q.add_argument("--rows", type=int, default=2)
    q.add_argument("--layout", choices=["yx", "xy"], default="yx")
    q = sub.add_parser("frame-features")
    q.add_argument("--manifest", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--ground-truth", action="store_true")
    q.add_argument("--bins", type=int, default=20)
    q.add_argument("--sigma", type=float, default=0)
    q.add_argument("--quality-config")
    q = sub.add_parser("pair-features")
    q.add_argument("--pairs", required=True)
    q.add_argument("--frames", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--ground-truth", action="store_true")
    q.add_argument("--bins", type=int, default=20)
    q.add_argument("--sigma", type=float, default=0)
    q.add_argument("--quality-config")
    q = sub.add_parser("temporal-features")
    q.add_argument("--frames", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--lags", nargs="+", type=int, default=[1])
    q.add_argument("--max-gap-hours", type=float, default=3)
    for name in ["train-seg", "train-cls"]:
        q = sub.add_parser(name)
        q.add_argument("--config", required=True)
        q.add_argument("--output", required=True)
        q.add_argument("--device", default="cuda")
    for name in ["eval-seg", "eval-cls"]:
        q = sub.add_parser(name)
        q.add_argument("--checkpoint", required=True)
        q.add_argument("--manifest" if name == "eval-seg" else "--features", required=True)
        q.add_argument("--output", required=True)
        q.add_argument("--device", default="cpu")
        q.add_argument("--split", choices=["train", "val", "test"], default="test")
        if name == "eval-seg":
            q.add_argument("--limit", type=int, default=0)
    q = sub.add_parser("benchmark")
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--image", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--device", default="cpu")
    q.add_argument("--warmup", type=int, default=3)
    q.add_argument("--repeats", type=int, default=10)
    q = sub.add_parser("infer-pairs")
    q.add_argument("--seg-checkpoint", required=True)
    q.add_argument("--cls-checkpoint", required=True)
    q.add_argument("--pairs", required=True)
    q.add_argument("--frames", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--device", default="cpu")
    q.add_argument("--split", choices=["val", "test"], default="test")
    q.add_argument("--warmup", type=int, default=2)
    q = sub.add_parser("replay")
    q.add_argument("--predictions", required=True)
    q.add_argument("--events", required=True)
    q.add_argument("--monitoring", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--consecutive", type=int, default=1)
    q.add_argument("--max-gap-hours", type=float, default=2)
    q.add_argument("--warning-hours", type=float, default=0)
    q = sub.add_parser("plot-classifier")
    q.add_argument("--predictions", required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("aggregate")
    q.add_argument("--runs", nargs="+", required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("plan")
    q.add_argument("--config", required=True)
    q.add_argument("--output", required=True)
    return p


def metadata_path(path):
    return resolve(path).with_suffix(".meta.json")


def copy_feature_metadata(input_path, output, **updates):
    src = metadata_path(input_path)
    meta = read_json(src) if src.exists() else {"source": "untracked"}
    meta.update(updates)
    write_json(metadata_path(output), meta)


def main(argv=None):
    a = parser().parse_args(argv)
    from . import data
    result = None
    if a.command == "doctor":
        from importlib.metadata import version, PackageNotFoundError
        result = {"python": sys.executable, "versions": {}}
        for name in ["numpy", "torch", "torchvision", "scikit-learn", "scipy", "Pillow",
                     "matplotlib", "transformers"]:
            try:
                result["versions"][name] = version(name)
            except PackageNotFoundError:
                result["versions"][name] = None
        import torch
        result["cuda_available"] = torch.cuda.is_available()
    elif a.command == "inventory":
        cfg = read_json(a.config)
        rows = []
        for spec in cfg["datasets"]:
            scanned = data.scan_dataset(spec["root"], spec["camera"], spec["mode"],
                                        a.hash_images, a.check_sizes)
            if spec.get("mask_mapping"):
                for row in scanned:
                    row["mask_mapping"] = spec["mask_mapping"]
            rows += scanned
        write_csv(a.output, rows)
        write_json(metadata_path(a.output), {"source": "inventory", "config": cfg,
                   "samples": len(rows), "parents": len({r["parent_id"] for r in rows}),
                   "checked_sizes": a.check_sizes, "hashed_images": a.hash_images})
        result = {"samples": len(rows)}
    elif a.command == "pair-inventory":
        rows = data.scan_pairs(a.root, a.camera, a.mode, a.label)
        write_csv(a.output, rows)
        result = {"pairs": len(rows), "labels": "unverified directory inventory"}
    elif a.command == "merge-manifests":
        rows = [r for path in a.inputs for r in read_csv(path)]
        if len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate IDs when merging; reconcile versions first")
        # Prediction provenance must match, otherwise one model's provenance
        # would silently be dropped. Manifests without sidecars are inventories.
        metas = [read_json(metadata_path(path)) if metadata_path(path).exists() else
                 {"source": "inventory"} for path in a.inputs]
        if all(m.get("source") == "inventory" for m in metas):
            meta = {"source": "inventory", "inputs": a.inputs}
        else:
            if any(m != metas[0] for m in metas):
                raise ValueError("Different metadata/provenance; reconcile before merging")
            meta = metas[0]
        write_csv(a.output, rows)
        write_json(metadata_path(a.output), meta)
        result = {"samples": len(rows)}
    elif a.command == "joint-split":
        frames, pairs = read_csv(a.frames), read_csv(a.pairs)
        combined = [dict(r, record_kind="frame") for r in frames]
        combined += [dict(r, record_kind="pair") for r in pairs]
        if a.events:
            data.attach_events(combined, read_csv(a.events))
        combined, _ = data.split_rows(combined, method=a.method, seed=a.seed)
        out = resolve(a.output)
        out.mkdir(parents=True, exist_ok=False)
        result = {}
        for kind, source in [("frame", a.frames), ("pair", a.pairs)]:
            selected = [{k: v for k, v in r.items() if k != "record_kind"}
                        for r in combined if r["record_kind"] == kind]
            path = out / (kind+"s.csv")
            write_csv(path, selected)
            copy_feature_metadata(source, path, split_method=a.method, seed=a.seed)
            result[kind] = data.validate_splits(selected, require_all=False)
        write_json(out / "audit.json", result)
    elif a.command == "split":
        rows = read_csv(a.manifest)
        if a.events:
            data.attach_events(rows, read_csv(a.events))
        rows, folds = data.split_rows(rows, a.train, a.val, a.seed, a.method, a.folds)
        out = resolve(a.output)
        out.mkdir(parents=True, exist_ok=False)
        write_csv(out / "manifest.csv", rows)
        copy_feature_metadata(a.manifest, out / "manifest.csv", split_method=a.method, seed=a.seed)
        for i, fold in enumerate(folds, 1):
            write_csv(out / "fold_{}.csv".format(i), fold)
            copy_feature_metadata(a.manifest, out / "fold_{}.csv".format(i),
                                  split_method="group_cross_validation", fold=i)
        result = data.validate_splits(rows)
        write_json(out / "audit.json", result)
    elif a.command == "audit-legacy":
        result = data.audit_legacy(a.voc)
        write_json(a.output, result)
    elif a.command == "check-manifest":
        rows = read_csv(a.manifest)
        result = data.validate_splits(rows)
        if rows[0].get("mask_path"):
            result["mask_samples"] = data.audit_masks(rows, a.mask_samples)
    elif a.command == "import-legacy":
        from .features import load_legacy, audit_legacy_features
        report = audit_legacy_features(a.folder)
        diagnostic = resolve(a.output).with_suffix(".audit.json")
        write_json(diagnostic, report)
        if not report["valid"]:
            raise ValueError("Legacy features are inconsistent; review " + str(diagnostic))
        rows = load_legacy(a.folder, a.camera)
        write_csv(a.output, rows)
        write_json(metadata_path(a.output), {
            "source": "legacy_unverified", "units": "legacy original units; not normalized",
            "label_status": "requires review", "segmentation_provenance": "unknown"})
        result = {"samples": len(rows)}
    elif a.command == "audit-legacy-features":
        from .features import audit_legacy_features
        result = audit_legacy_features(a.folder, a.source_pairs)
        write_json(a.output, result)
        if a.rows_output:
            write_csv(a.rows_output, result["rows"])
        result = {k: v for k, v in result.items() if k not in {"issues", "rows"}}
    elif a.command == "reconstruct":
        from .reconstruct import reconstruct
        result = reconstruct(read_csv(a.manifest), a.output, a.columns, a.rows, a.layout)
    elif a.command in {"frame-features", "pair-features"}:
        from PIL import Image
        from .features import geometry, image_quality, extract_pair_features
        from .masks import read_mask
        source = a.manifest if a.command == "frame-features" else a.frames
        quality = read_json(a.quality_config) if a.quality_config else {}
        frames = read_csv(source)
        if a.command == "pair-features":
            rows = extract_pair_features(read_csv(a.pairs), frames, a.bins, a.sigma,
                                         a.ground_truth, quality)
        else:
            if len({r["parent_id"] for r in frames}) != len(frames):
                raise ValueError("Geometry requires reconstructed/full frames, not individual patches")
            rows = []
            for r in frames:
                g = geometry(read_mask(r, predicted=not a.ground_truth), a.bins, a.sigma)
                g["edge_coverage"] = sum(v for k, v in g.items() if k.startswith("coverage_")) / a.bins
                with Image.open(resolve(r["image_path"])) as im:
                    q = image_quality(im.convert("RGB"), **quality)
                rows.append(dict(r, **g, **q))
        write_csv(a.output, rows)
        copy_feature_metadata(source, a.output, source="ground_truth" if a.ground_truth else "predicted",
                              bins=a.bins, sigma=a.sigma, quality_config=quality, feature_kind=a.command)
        result = {"samples": len(rows)}
    elif a.command == "temporal-features":
        from .features import temporal_features
        rows = temporal_features(read_csv(a.frames), a.lags, a.max_gap_hours)
        if not rows:
            raise ValueError("No causal windows; check timestamps, groups and gap limits")
        write_csv(a.output, rows)
        copy_feature_metadata(a.frames, a.output, lags=a.lags, max_gap_hours=a.max_gap_hours)
        result = {"samples": len(rows)}
    elif a.command == "train-seg":
        from .segmentation import train
        result = str(train(read_json(a.config), a.output, a.device))
    elif a.command == "eval-seg":
        from .segmentation import evaluate
        result = evaluate(a.checkpoint, a.manifest, a.output, a.device, a.split, a.limit)
    elif a.command == "train-cls":
        from .classification import fit
        result = str(fit(read_json(a.config), a.output, a.device))
    elif a.command == "eval-cls":
        from .classification import evaluate
        result = evaluate(a.checkpoint, a.features, a.output, a.device, a.split)
    elif a.command == "benchmark":
        from .segmentation import benchmark
        result = benchmark(a.checkpoint, a.image, a.output, a.device, a.warmup, a.repeats)
    elif a.command == "infer-pairs":
        from .pipeline import infer_pairs
        result = infer_pairs(a.seg_checkpoint, a.cls_checkpoint, a.pairs, a.frames,
                             a.output, a.device, a.split, a.warmup)
    elif a.command == "replay":
        from .replay import run
        result = run(a.predictions, a.events, a.monitoring, a.output,
                     consecutive=a.consecutive, max_gap_hours=a.max_gap_hours, warning_hours=a.warning_hours)
    elif a.command == "plot-classifier":
        from .reporting import classifier_figures
        classifier_figures(a.predictions, a.output)
    elif a.command == "aggregate":
        from .reporting import aggregate
        result = aggregate(a.runs, a.output)
    elif a.command == "plan":
        from .suite import plan
        result = plan(read_json(a.config), a.output)
    if result is not None:
        print(json.dumps(clean_json(result), ensure_ascii=False, indent=2))
    return 0
