"""Small-data baselines. Fit preprocessing on train, select on val, test once."""
import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn

from .data import ensure_unseen, provenance, validate_splits
from .io import (completed, device_for, digest, new_run, read_csv, read_json,
                 resolve, seed_all, write_csv, write_json)
from .metrics import classification, choose_threshold, group_bootstrap
from .models import GeometryCNN


class TrainOnlyScaler:
    """Keep all-missing columns so geometry branch dimensions cannot change."""
    def fit(self, x):
        self.median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0
                                for c in x.T])
        filled = np.where(np.isfinite(x), x, self.median)
        self.mean = filled.mean(0)
        self.scale = filled.std(0)
        self.scale[self.scale < 1e-8] = 1
        return self

    def transform(self, x):
        return ((np.where(np.isfinite(x), x, self.median)-self.mean)/self.scale).astype(np.float32)


def feature_columns(rows, feature_set):
    area = sorted(k for k in rows[0] if k.startswith("area_"))
    upper = sorted(k for k in rows[0] if k.startswith("upper_") and "_abs_" not in k)
    lower = sorted(k for k in rows[0] if k.startswith("lower_") and "_abs_" not in k)
    extra = sorted(k for k in rows[0] if "_abs_" in k or k.startswith("width_"))
    choices = {"area": area, "boundary": upper+lower, "both": area+upper+lower,
               "geometry": area+upper+lower+extra}
    if feature_set not in choices or not choices[feature_set]:
        raise ValueError("Missing/unknown feature set {}".format(feature_set))
    return choices[feature_set]


def matrix(rows, columns):
    return np.array([[float("nan" if r.get(c) in (None, "") else r[c])
                      for c in columns] for r in rows], dtype=float)


def labels(rows):
    if any(str(r.get("label", "")) not in {"0", "1"} for r in rows):
        raise ValueError("Classification needs reviewed binary labels, not empty labels")
    return np.array([int(r["label"]) for r in rows])


def accepted(rows, config):
    if not config.get("quality_gate", False):
        return np.ones(len(rows), dtype=bool)
    return np.array([str(r.get("quality_valid", "0")) == "1" and
                     float(r.get("edge_coverage", 1)) >= config.get("edge_coverage_min", 0)
                     for r in rows])


def upstream_metadata(features):
    path = resolve(features).with_suffix(".meta.json")
    return read_json(path) if path.exists() else {"source": "untracked"}


def check_upstream(rows, meta):
    if meta.get("source") == "predicted":
        if "provenance" not in meta:
            raise ValueError("Predicted features need segmentation provenance")
        ensure_unseen(rows, meta["provenance"])


def neural_model(kind, columns, hidden=32):
    if kind == "dual_cnn":
        n_area = sum(c.startswith("area_") for c in columns)
        return GeometryCNN(n_area, len(columns)-n_area, hidden)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(len(columns), hidden), nn.ReLU(), nn.Dropout(.1),
                             nn.Linear(hidden, 2))
    raise ValueError("Unknown neural classifier")


def fit(config, output, device="cuda"):
    if int(config.get("epochs", 100)) < 1:
        raise ValueError("epochs must be positive")
    rows = read_csv(config["features"])
    validate_splits(rows)
    labels(rows)
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    meta = upstream_metadata(config["features"])
    unreviewed = any("unverified" in r.get("label_source", "") or
                     not r.get("label_source") for r in rows)
    if (unreviewed or meta.get("source") in {"untracked", "legacy_unverified"}) and not config.get("allow_unverified_labels", False):
        raise ValueError("Review label_source and feature provenance, or explicitly set "
                         "allow_unverified_labels=true for exploratory runs only")
    # A segmentation model must not have seen final test dates/events.
    check_upstream([r for r in rows if r["split"] == "test"], meta)
    # Validation for the classifier is also held out from segmentation training.
    if meta.get("source") == "predicted":
        ensure_unseen(va, meta["provenance"], include_validation=False)
    tr = [r for r, ok in zip(tr, accepted(tr, config)) if ok]
    va = [r for r, ok in zip(va, accepted(va, config)) if ok]
    if set(labels(tr)) != {0, 1} or set(labels(va)) != {0, 1}:
        raise ValueError("Train and validation both need both classes after quality gating")
    cols = feature_columns(tr, config.get("feature_set", "both"))
    xtr, xva = matrix(tr, cols), matrix(va, cols)
    scaler = TrainOnlyScaler().fit(xtr)
    xtr, xva = scaler.transform(xtr), scaler.transform(xva)
    ytr, yva = labels(tr), labels(va)
    seed = int(config.get("seed", 42))
    seed_all(seed, int(config.get("threads", 2)))
    run = new_run(output, config, [config["features"]])
    kind = config.get("model", "logistic")
    payload = {"kind": kind, "columns": cols, "scaler": scaler,
               "config": config, "provenance": provenance(rows), "upstream": meta}
    if kind in {"logistic", "forest"}:
        model = (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
                 if kind == "logistic" else RandomForestClassifier(
                     n_estimators=config.get("trees", 200), max_depth=config.get("max_depth", 6),
                     class_weight="balanced", random_state=seed, n_jobs=config.get("threads", 2)))
        model.fit(xtr, ytr)
        val_scores = model.predict_proba(xva)[:, 1]
        payload["model"] = model
    elif kind == "area_threshold":
        if any(not c.startswith("area_") for c in cols):
            raise ValueError("area_threshold requires feature_set=area")
        # Magnitude of raw normalized area change; validation selects threshold.
        val_scores = np.nanmean(np.abs(matrix(va, cols)), axis=1)
        if not np.isfinite(val_scores).all():
            raise ValueError("Missing area features for threshold baseline")
    elif kind in {"dual_cnn", "mlp"}:
        dev = device_for(device)
        hidden = config.get("hidden", 32)
        model = neural_model(kind, cols, hidden).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", .001))
        weights = torch.tensor(len(ytr)/(2*np.bincount(ytr, minlength=2)), dtype=torch.float32, device=dev)
        criterion = nn.CrossEntropyLoss(weight=weights)
        best, stale, history = float("inf"), 0, []
        generator = np.random.default_rng(seed)
        for epoch in range(config.get("epochs", 100)):
            model.train()  # Restore dropout every epoch.
            order = generator.permutation(len(tr))
            for start in range(0, len(order), config.get("batch_size", 32)):
                idx = order[start:start+config.get("batch_size", 32)]
                x = torch.tensor(xtr[idx], device=dev)
                y = torch.tensor(ytr[idx], device=dev)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                logits = model(torch.tensor(xva, device=dev))
                val_loss = float(criterion(logits, torch.tensor(yva, device=dev)))
            history.append({"epoch": epoch+1, "val_loss": val_loss})
            if val_loss < best:
                best, stale = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= config.get("patience", 20):
                break
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            val_scores = model(torch.tensor(xva, device=dev)).softmax(1)[:, 1].cpu().numpy()
        payload.update(state_dict=best_state, hidden=hidden)
        write_csv(run / "history.csv", history)
    else:
        raise ValueError("Unknown classifier {}".format(kind))
    threshold = float(config["threshold"] if "threshold" in config else
                      choose_threshold(yva, val_scores))
    payload["threshold"] = threshold
    joblib.dump(payload, run / "model.joblib")
    write_json(run / "validation_metrics.json", classification(yva, val_scores, threshold))
    write_csv(run / "validation_predictions.csv", [
        dict(sample_id=r["sample_id"], y_true=int(y), score=float(s))
        for r, y, s in zip(va, yva, val_scores)])
    completed(run)
    return run


class Predictor:
    """Load a neural classifier once for repeated per-pair inference."""
    def __init__(self, payload, device="cpu"):
        self.payload, self.device, self.model = payload, device, None
        if payload["kind"] in {"dual_cnn", "mlp"}:
            self.device = device_for(device)
            self.model = neural_model(payload["kind"], payload["columns"], payload["hidden"]).to(self.device)
            self.model.load_state_dict(payload["state_dict"])
            self.model.eval()

    def __call__(self, rows):
        return predict(self.payload, rows, self.device, self.model)


def predict(payload, rows, device="cpu", model=None):
    cols, kind = payload["columns"], payload["kind"]
    if any(c not in rows[0] for c in cols):
        raise ValueError("Feature schema differs from classifier training")
    x = matrix(rows, cols)
    if kind == "area_threshold":
        scores = np.nanmean(np.abs(x), axis=1)
    elif kind in {"logistic", "forest"}:
        scores = payload["model"].predict_proba(payload["scaler"].transform(x))[:, 1]
    else:
        dev = device_for(device)
        if model is None:
            model = neural_model(kind, cols, payload["hidden"]).to(dev)
            model.load_state_dict(payload["state_dict"])
            model.eval()
        with torch.inference_mode():
            scores = model(torch.tensor(payload["scaler"].transform(x), device=dev)).softmax(1)[:, 1].cpu().numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite classifier scores")
    return scores


def evaluate(checkpoint, features, output, device="cpu", split="test"):
    # joblib deserializes code: only load checkpoints produced by a trusted run.
    payload = joblib.load(resolve(checkpoint))
    all_rows = read_csv(features)
    validate_splits(all_rows)
    rows = [r for r in all_rows if r["split"] == split]
    if not rows:
        raise ValueError("Empty evaluation split")
    meta = upstream_metadata(features)
    for key in ["source", "bins", "sigma", "quality_config", "feature_kind",
                "lags", "max_gap_hours", "checkpoint_sha256"]:
        if meta.get(key) != payload["upstream"].get(key):
            raise ValueError("Feature extraction differs from classifier training: " + key)
    if split == "test":
        ensure_unseen(rows, payload["provenance"])
        check_upstream(rows, upstream_metadata(features))
        check_upstream(rows, payload["upstream"])
    elif split == "val":
        ensure_unseen(rows, payload["provenance"], include_validation=False)
    scores = predict(payload, rows, device)
    valid = accepted(rows, payload["config"])
    y = labels(rows)
    threshold = payload["threshold"]
    # For overall alert performance, rejected observations produce no alert.
    effective = np.where(valid, scores, threshold - max(1, abs(threshold)))
    metrics = classification(y, effective, threshold)
    metrics.update(coverage=float(valid.mean()), rejected=int((~valid).sum()),
                   rejected_positives=int(y[~valid].sum()),
                   score_source=payload["upstream"].get("source", "untracked"))
    metrics["selective_metrics"] = classification(y[valid], scores[valid], threshold) if valid.any() else None
    metrics["uncertainty"] = group_bootstrap(y, effective, [r["group_id"] for r in rows],
                                             threshold, repeats=payload["config"].get("bootstrap", 1000))
    cohorts = {}
    for r in rows:
        r["month"] = r["timestamp"][:7]
    for field in ["mode", "weather", "month"]:
        for value in sorted({r.get(field, "unknown") for r in rows}):
            indices = [i for i, r in enumerate(rows) if r.get(field, "unknown") == value]
            cohorts[field+":"+value] = dict(classification(y[indices], effective[indices], threshold),
                coverage=float(valid[indices].mean()),
                independent_groups=len({rows[i]["group_id"] for i in indices}))
    run = new_run(output, {"split": split, "threshold": threshold}, [features, checkpoint])
    output_rows = [dict(r, y_true=int(label), score=float(score), threshold=threshold,
                        valid=int(ok), y_pred=int(ok and score >= threshold))
                   for r, label, score, ok in zip(rows, y, scores, valid)]
    write_csv(run / "predictions.csv", output_rows)
    write_json(run / "metrics.json", metrics)
    write_json(run / "cohorts.json", cohorts)
    completed(run)
    return metrics
