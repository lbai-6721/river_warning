"""Loopback HTTP service, bounded jobs, validated uploads and persistent history."""
import base64
import csv
import io
import json
import mimetypes
import secrets
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

from ..io import clean_json, resolve
from .engine import AlertState, Catalog, Engine, iso_time, validate_times


WEB = Path(__file__).parent / "web"


def validate_alert(options):
    keys = {"threshold", "clear_threshold", "consecutive", "clear_consecutive", "max_gap_seconds"}
    if set(options) != keys:
        raise ValueError("告警设置字段不完整")
    if not 0 <= options["clear_threshold"] < options["threshold"] <= 1:
        raise ValueError("解除阈值应小于触发阈值，且都位于 0–1")
    for key in ("consecutive", "clear_consecutive"):
        if type(options[key]) is not int or not 1 <= options[key] <= 100:
            raise ValueError("连续确认次数应为 1–100 的整数")
    if not 1 <= options["max_gap_seconds"] <= 86400:
        raise ValueError("最大观测间隔应为 1–86400 秒")
    return options


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "monitor.sqlite3"
        with sqlite3.connect(self.db) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS results (id TEXT PRIMARY KEY, created TEXT, data TEXT)")

    def add(self, result):
        result = clean_json(result)
        with sqlite3.connect(self.db) as conn:
            conn.execute("INSERT INTO results VALUES (?, ?, ?)",
                         (result["result_id"], result["created_at"], json.dumps(result, ensure_ascii=False)))

    def history(self, limit=300):
        with sqlite3.connect(self.db) as conn:
            rows = conn.execute("SELECT data FROM results ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]


class Monitor:
    def __init__(self, config):
        self.config = config
        validate_alert(config["alert"])
        if config["host"] != "127.0.0.1":
            raise ValueError("此本地版本仅允许绑定 127.0.0.1")
        self.store = Store(resolve(config["output"]))
        settings = self.store.root / "alert_settings.json"
        if settings.exists():
            config["alert"] = validate_alert(json.loads(settings.read_text(encoding="utf-8")))
        self.catalog = Catalog(config)
        for image in (self.store.root / "uploads").glob("*/*.image"):
            self.catalog.register(image)
        self.engine = Engine(config, self.catalog)
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="river-inference")
        self.jobs = {}
        self.active = None
        self.token = secrets.token_urlsafe(32)

    def bootstrap(self):
        profiles = []
        for key, value in self.config["profiles"].items():
            profiles.append(dict(id=key, name=value["name"],
                                 ready=resolve(value["checkpoint"]).is_file(),
                                 checkpoint=Path(value["checkpoint"]).name,
                                 classifier=bool(value.get("classifier"))))
        with self.lock:
            return dict(profiles=profiles, pairs=self.catalog.pairs, token=self.token,
                        alert=self.config["alert"], device=str(self.engine.device),
                        active_job=self.active, legacy_pairs=len(self.catalog.legacy),
                        legacy_issue=self.catalog.legacy_issue,
                        legacy_ready=resolve(self.config["legacy_checkpoint"]).is_file())

    def change_settings(self, options):
        validate_alert(options)
        with self.lock:
            if self.active:
                raise ValueError("请先停止当前任务，再修改参数")
            self.config["alert"] = dict(options)
            path = self.store.root / "alert_settings.json"
            path.write_text(json.dumps(options, ensure_ascii=False, indent=2), encoding="utf-8")
        return options

    def upload(self, body):
        with self.lock:
            if self.active:
                raise ValueError("已有任务正在运行，请等待完成或停止")
        items = body.get("files", [])
        mode = body.get("mode")
        if mode not in self.config["profiles"] or not 2 <= len(items) <= 20:
            raise ValueError("请选择日夜模式，并上传 2–20 张图片")
        frames = []
        # Validate the entire request before writing files.
        for item in items:
            when = iso_time(item["timestamp"])
            data = base64.b64decode(item["data"], validate=True)
            if len(data) > 12*1024*1024:
                raise ValueError("单张图片不能超过 12 MB")
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > 24_000_000 or min(image.size) < 32:
                    raise ValueError("图像尺寸应至少为 32 像素，且不超过 2400 万像素")
                image.verify()
            frames.append((when, data, Path(item.get("name", "image")).name))
        frames.sort(key=lambda v: v[0])
        for a, b in zip(frames, frames[1:]):
            validate_times(a[0], b[0])
        batch_id = uuid.uuid4().hex
        folder = self.store.root / "uploads" / batch_id
        folder.mkdir(parents=True)
        registered = []
        with self.lock:
            for index, (when, data, name) in enumerate(frames):
                path = folder / (str(index) + ".image")
                path.write_bytes(data)
                registered.append(dict(id=self.catalog.register(path), timestamp=when, name=name))
        # The first image is a fixed operator-selected reference for this session.
        pairs = [dict(id="upload-" + batch_id + "-" + str(i), name=cur["name"], mode=mode,
                      camera_id="upload-" + batch_id, reference=registered[0]["id"], current=cur["id"],
                      timestamp_a=registered[0]["timestamp"], timestamp=cur["timestamp"], legacy_available=False)
                 for i, cur in enumerate(registered[1:], 1)]
        return self.start(pairs, "sequence" if len(pairs) > 1 else "single")

    def start_catalog(self, body):
        ids = body.get("pair_ids", [])
        if not 1 <= len(ids) <= 100 or len(set(ids)) != len(ids):
            raise ValueError("请选择 1–100 个不同图像对")
        try:
            pairs = [dict(self.catalog.lookup[key]) for key in ids]
        except KeyError:
            raise ValueError("图像对不存在")
        return self.start(pairs, "single" if len(pairs) == 1 else "samples")

    def start(self, pairs, kind):
        with self.lock:
            if self.active:
                raise ValueError("已有任务正在运行，请等待完成或停止")
            job_id = uuid.uuid4().hex
            job = dict(id=job_id, kind=kind, status="queued", total=len(pairs), done=0,
                       message="等待推理", results=[], errors=[], stop=threading.Event())
            self.jobs[job_id] = job
            self.active = job_id
            while len(self.jobs) > 30:
                self.jobs.pop(next(iter(self.jobs)))
            self.pool.submit(self._run, job, pairs)
            return {"job_id": job_id}

    def _run(self, job, pairs):
        with self.lock:
            job["status"] = "running"
        states = {}
        try:
            for pair in pairs:
                if job["stop"].is_set():
                    break
                with self.lock:
                    job["message"] = "正在分析：" + pair["name"]
                result_id = uuid.uuid4().hex
                try:
                    start = time.perf_counter()
                    result = self.engine.analyze(pair, self.store.root / "results" / result_id)
                    options = dict(self.config["alert"])
                    options["threshold"] = result["threshold"]
                    if options["clear_threshold"] >= options["threshold"]:
                        options["clear_threshold"] = options["threshold"] * .7
                    if job["kind"] == "sequence":
                        state = states.setdefault(pair["camera_id"], AlertState(options))
                        alarm = state.update(result["score"], pair["timestamp"], result["prediction"] is not None)
                    else:
                        # Independent sample pairs are not a continuous camera stream.
                        alarm = "candidate" if result["prediction"] == 1 else (
                            "idle" if result["prediction"] == 0 else "unknown")
                    result.update(result_id=result_id, created_at=datetime.now().isoformat(),
                                  alarm=alarm, seconds=round(time.perf_counter()-start, 3),
                                  session_id=job["id"], session_kind=job["kind"], alert_config=options,
                                  profile=self.config["profiles"][pair["mode"]],
                                  quality_config=self.config["quality"])
                    self.store.add(result)
                    with self.lock:
                        job["results"].append(clean_json(result))
                except Exception as exc:
                    if job["kind"] == "sequence" and pair["camera_id"] in states:
                        state = states[pair["camera_id"]]
                        if state.last != datetime.fromisoformat(pair["timestamp"]):
                            state.update(None, pair["timestamp"], False)
                        else:
                            state.high = state.low = 0
                    with self.lock:
                        job["errors"].append(dict(pair_id=pair["id"], name=pair["name"], message=str(exc)))
                with self.lock:
                    job["done"] += 1
            with self.lock:
                job["status"] = "cancelled" if job["stop"].is_set() else (
                    "partial" if job["errors"] and job["results"] else "failed" if job["errors"] else "complete")
                job["message"] = "已停止" if job["stop"].is_set() else "处理结束"
        except Exception as exc:
            with self.lock:
                job["status"] = "failed"
                job["errors"].append({"message": str(exc)})
        finally:
            with self.lock:
                self.active = None

    def job(self, job_id, stop=False):
        with self.lock:
            if job_id not in self.jobs:
                raise ValueError("任务不存在或服务已重启，请查看历史记录")
            job = self.jobs[job_id]
            if stop:
                job["stop"].set()
            return {k: v for k, v in job.items() if k != "stop"}


def handler_for(monitor):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, data, content_type="application/json; charset=utf-8", status=200):
            if not isinstance(data, bytes):
                data = json.dumps(clean_json(data), ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def guard(self, mutation=False):
            expected = "127.0.0.1:" + str(self.server.server_port)
            if self.headers.get("Host") != expected:
                raise PermissionError("仅接受本地访问")
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + expected:
                raise PermissionError("不接受跨站请求")
            if mutation and self.headers.get("X-Monitor-Token") != monitor.token:
                raise PermissionError("请刷新页面后重试")

        def do_GET(self):
            try:
                self.guard()
                parsed = urlparse(self.path)
                route = parsed.path
                if route == "/api/bootstrap":
                    return self.send(monitor.bootstrap())
                if route == "/api/history":
                    return self.send(monitor.store.history())
                if route == "/api/job":
                    return self.send(monitor.job(parse_qs(parsed.query).get("id", [""])[0]))
                if route == "/api/export":
                    stream = io.StringIO(newline="")
                    fields = ["result_id", "timestamp", "mode", "prediction", "score", "threshold", "source", "alarm", "area_change_pp", "seconds"]
                    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(monitor.store.history(100000))
                    return self.send(stream.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8")
                if route.startswith("/api/image/"):
                    key = route.rsplit("/", 1)[-1]
                    path = monitor.catalog.files.get(key)
                    if path is None:
                        return self.send({"error": "图像不存在"}, status=404)
                    with Image.open(path) as im:
                        im = im.convert("RGB")
                        im.thumbnail((1280, 800))
                        buf = io.BytesIO()
                        im.save(buf, format="JPEG", quality=85)
                    return self.send(buf.getvalue(), "image/jpeg")
                if route.startswith("/results/"):
                    parts = route.split("/")
                    if (len(parts) != 4 or len(parts[2]) != 32 or any(c not in "0123456789abcdef" for c in parts[2])
                            or parts[3] not in {"reference.jpg", "current.jpg", "change.jpg", "reference_mask.png", "current_mask.png"}):
                        raise ValueError("无效的结果路径")
                    path = monitor.store.root / "results" / parts[2] / parts[3]
                else:
                    assets = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
                    if route not in assets:
                        return self.send({"error": "页面不存在"}, status=404)
                    path = WEB / assets[route]
                if not path.is_file():
                    return self.send({"error": "文件不存在"}, status=404)
                mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                return self.send(path.read_bytes(), mime + ("; charset=utf-8" if mime.startswith("text/") else ""))
            except PermissionError as exc:
                self.send({"error": str(exc)}, status=403)
            except (ValueError, OSError) as exc:
                self.send({"error": str(exc)}, status=400)

        def do_POST(self):
            try:
                self.guard(True)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 64*1024*1024:
                    raise ValueError("请求为空或超过 64 MB")
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("请求必须是 JSON 对象")
                route = urlparse(self.path).path
                if route == "/api/run":
                    result = monitor.start_catalog(body)
                elif route == "/api/upload":
                    result = monitor.upload(body)
                elif route == "/api/stop":
                    result = monitor.job(body["job_id"], True)
                elif route == "/api/settings":
                    result = monitor.change_settings(body)
                else:
                    return self.send({"error": "接口不存在"}, status=404)
                self.send(result)
            except PermissionError as exc:
                self.send({"error": str(exc)}, status=403)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.send({"error": str(exc)}, status=400)

    return Handler


def serve(config):
    monitor = Monitor(config)
    server = ThreadingHTTPServer((config["host"], config["port"]), handler_for(monitor))
    server.daemon_threads = True
    print("River Monitor: http://127.0.0.1:{}  ({} pairs)".format(server.server_port, len(monitor.catalog.pairs)), flush=True)
    try:
        server.serve_forever()
    finally:
        if monitor.active:
            monitor.job(monitor.active, True)
        server.server_close()
        monitor.pool.shutdown(wait=True)
