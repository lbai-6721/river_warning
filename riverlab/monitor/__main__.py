import argparse
import json

from ..io import resolve
from .service import serve


def main():
    parser = argparse.ArgumentParser(description="本地河流堵塞监控界面")
    parser.add_argument("--config", default="configs/monitor.json")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    config = json.loads(resolve(args.config).read_text(encoding="utf-8-sig"))
    if args.port is not None:
        config["port"] = args.port
    serve(config)


if __name__ == "__main__":
    main()
