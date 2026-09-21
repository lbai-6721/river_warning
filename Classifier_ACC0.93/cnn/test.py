"""Compatibility entry point for leakage-checked held-out evaluation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from riverlab.cli import main
    raise SystemExit(main(["eval-cls"] + sys.argv[1:]))
