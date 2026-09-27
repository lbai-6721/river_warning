#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${RIVER_PYTHON:-python}"
day_session="${DAY_TMUX_SESSION:-river-day-full-es}"

while tmux has-session -t "$day_session" 2>/dev/null; do
  echo "waiting for daytime queue: $day_session"
  sleep 60
done

"$python_bin" - <<'PY'
import json
from pathlib import Path
names = ["day_full_scratch_early_stop_{:03d}".format(i) for i in range(3)]
for name in names:
    path = Path("runs") / name / "run.json"
    if not path.exists() or json.loads(path.read_text(encoding="utf-8"))["status"] != "complete":
        raise SystemExit("Daytime queue did not complete: " + name)
print("daytime queue complete")
PY

inventory="datasets/segmentation/night/source_all_inventory.csv"
split_dir="datasets/segmentation/night/source_all_split"
reconstructed="datasets/segmentation/night/reconstructed_roi"
if [[ ! -f "$split_dir/manifest.csv" ]]; then
  "$python_bin" -m riverlab inventory \
    --config configs/night_patch_dataset.json \
    --output "$inventory" --check-sizes
  "$python_bin" -m riverlab split \
    --manifest "$inventory" --output "$split_dir" \
    --method time --train 0.6 --val 0.2 --seed 42
fi
"$python_bin" -m riverlab check-manifest \
  --manifest "$split_dir/manifest.csv" --mask-samples 100
if [[ ! -d "$reconstructed" ]]; then
  "$python_bin" -m riverlab reconstruct \
    --manifest "$split_dir/manifest.csv" \
    --output "$reconstructed" --columns 5 --rows 2 --layout yx
fi
"$python_bin" -m riverlab check-manifest \
  --manifest "$reconstructed/manifest.csv" --mask-samples 100

plan="artifacts/night_reconstructed_early_stop_plan"
if [[ ! -d "$plan" ]]; then
  "$python_bin" -m riverlab plan \
    --config configs/segmentation_night_reconstructed_early_stop.json \
    --output "$plan"
fi
"$python_bin" scripts/run_segmentation_comparison.py "$plan" \
  --logs artifacts/night_reconstructed_early_stop_logs
