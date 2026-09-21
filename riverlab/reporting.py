"""Figures and summary tables from saved real predictions; no invented values."""
from collections import defaultdict
from datetime import datetime

import numpy as np

from .io import read_csv, read_json, resolve, write_csv, write_json


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def timeline(rows, events, output):
    plt = pyplot()
    cameras = sorted({r["camera_id"] for r in rows})
    fig, axes = plt.subplots(len(cameras), 1, figsize=(11, max(3, len(cameras)*3)), squeeze=False)
    for camera, ax in zip(cameras, axes[:, 0]):
        stream = sorted([r for r in rows if r["camera_id"] == camera], key=lambda r: r["timestamp"])
        times = [datetime.fromisoformat(r["timestamp"]) for r in stream]
        scores = [float(r["score"]) if str(r.get("valid", 1)) == "1" else np.nan for r in stream]
        ax.plot(times, scores, ".-", label="score")
        ax.plot(times, [float(r["threshold"]) for r in stream], "--", label="fixed threshold")
        for e in events:
            if e["camera_id"] == camera:
                ax.axvspan(datetime.fromisoformat(e["start"]), datetime.fromisoformat(e["end"]),
                           color="red", alpha=.15)
        ax.set_title(camera)
        ax.set_ylabel("Score (not calibrated probability)")
        ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(resolve(output), dpi=160)
    plt.close(fig)


def classifier_figures(predictions, output):
    from sklearn.metrics import confusion_matrix, precision_recall_curve
    rows = read_csv(predictions)
    directory = resolve(output)
    directory.mkdir(parents=True, exist_ok=False)
    y = np.array([int(r["y_true"]) for r in rows])
    p = np.array([int(r["y_pred"]) for r in rows])
    scores = np.array([float(r["score"]) for r in rows])
    valid = np.array([str(r.get("valid", 1)) == "1" for r in rows])
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(4, 4))
    matrix = confusion_matrix(y, p, labels=[0, 1])
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="True")
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    if len(set(y[valid])) == 2:
        precision, recall, _ = precision_recall_curve(y[valid], scores[valid])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall, precision)
        ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1),
               title="Accepted observations only; coverage={:.1%}".format(valid.mean()))
        fig.tight_layout()
        fig.savefig(directory / "precision_recall.png", dpi=180)
        plt.close(fig)
    write_json(directory / "source.json", {"predictions": str(resolve(predictions)),
                                           "n": len(rows), "coverage": valid.mean()})


def aggregate(run_dirs, output):
    rows = []
    for directory in run_dirs:
        d = resolve(directory)
        m, run = read_json(d / "metrics.json"), read_json(d / "run.json")
        row = {"run": str(directory), "git_commit": run.get("git_commit", "")}
        row.update({k: v for k, v in m.items() if isinstance(v, (int, float, str)) or v is None})
        rows.append(row)
    write_csv(output, rows)
    return rows
