#!/usr/bin/env python3
"""Run a planned segmentation matrix sequentially and evaluate every checkpoint."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def completed(path):
    metadata = path / "run.json"
    if not metadata.exists():
        return False
    return json.loads(metadata.read_text(encoding="utf-8")).get("status") == "complete"


def execute(command, log_path, dry_run=False):
    print("$ " + " ".join(str(item) for item in command), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise SystemExit("Command failed with exit code {}: {}".format(
            code, " ".join(str(item) for item in command)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", help="Plan directory containing jobs.json")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--logs", default="artifacts/segmentation_comparison_logs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    plan = Path(args.plan)
    jobs = json.loads((plan / "jobs.json").read_text(encoding="utf-8"))
    runs, logs = Path(args.runs), Path(args.logs)
    for job in jobs:
        name = job["name"]
        train_output = runs / name
        if completed(train_output):
            print("skip completed training: " + name, flush=True)
        else:
            if train_output.exists() and not args.dry_run:
                raise SystemExit("Incomplete run already exists: {}".format(train_output))
            command = list(job["command"])
            command[0] = sys.executable
            command[command.index("--output") + 1] = str(train_output)
            command[command.index("--device") + 1] = args.device
            execute(command, logs / (name + ".train.log"), args.dry_run)

        if args.no_eval:
            continue
        eval_output = runs / (name + "_eval")
        if completed(eval_output):
            print("skip completed evaluation: " + name, flush=True)
            continue
        if eval_output.exists() and not args.dry_run:
            raise SystemExit("Incomplete evaluation already exists: {}".format(eval_output))
        checkpoint = train_output / "best.pt"
        command = [
            sys.executable, "-m", "riverlab", "eval-seg",
            "--checkpoint", str(checkpoint),
            "--manifest", job["config"]["manifest"],
            "--output", str(eval_output),
            "--device", args.device,
            "--split", "test",
        ]
        execute(command, logs / (name + ".eval.log"), args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
