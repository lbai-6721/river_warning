"""CPU tests for monitor state, provenance, job boundaries and local HTTP APIs."""
import base64
import copy
import csv
import io
import json
import sqlite3
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from PIL import Image
import joblib
import numpy as np

from riverlab.monitor.engine import AlertState, Catalog, validate_times
from riverlab.monitor.service import Monitor, Store, handler_for, validate_alert


ALERT = dict(threshold=.7, clear_threshold=.3, consecutive=2, clear_consecutive=2, max_gap_seconds=120)


class AlertTests(unittest.TestCase):
    def test_unknown_and_gap_preserve_alert_but_reset_confirmation(self):
        state = AlertState(ALERT)
        self.assertEqual(state.update(.9, "2026-01-01T00:00:00"), "pending")
        self.assertEqual(state.update(.9, "2026-01-01T00:01:00"), "triggered")
        self.assertEqual(state.update(None, "2026-01-01T00:02:00"), "interrupted")
        self.assertEqual(state.update(.1, "2026-01-01T00:03:00"), "active")
        self.assertEqual(state.update(.1, "2026-01-01T00:10:00"), "active")
        self.assertEqual(state.update(.1, "2026-01-01T00:11:00"), "cleared")

    def test_confirmation_resets_on_invalid_and_medium_observation(self):
        state = AlertState(ALERT)
        self.assertEqual(state.update(.9, "2026-01-01T00:00:00"), "pending")
        self.assertEqual(state.update(None, "2026-01-01T00:01:00"), "unknown")
        self.assertEqual(state.update(.9, "2026-01-01T00:02:00"), "pending")
        self.assertEqual(state.update(.5, "2026-01-01T00:03:00"), "idle")
        self.assertEqual(state.update(.9, "2026-01-01T00:04:00"), "pending")
        with self.assertRaises(ValueError):
            state.update(.9, "2026-01-01T00:04:00")

    def test_invalid_settings_and_time(self):
        for changed in ({"threshold":.1}, {"consecutive":1.5}, {"max_gap_seconds":0}):
            with self.assertRaises(ValueError):
                validate_alert(dict(ALERT, **changed))
        with self.assertRaises(ValueError):
            validate_times("2026-01-01T01:00:00", "2026-01-01T00:00:00")
        with self.assertRaises(ValueError):
            validate_times("2026-01-01T00:00:00", "2026-01-01T01:00:00+08:00")


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        names = ["20260101000000.jpg", "20260101010000.jpg", "20260101020000.jpg"]
        folder = self.root / "dataset/day/pairs/2026.1.1.0_2026.1.1.1"
        folder.mkdir(parents=True)
        for name in names:
            Image.new("RGB", (64, 64), (50, 120, 100)).save(folder / name)
        legacy = self.root / "legacy"
        legacy.mkdir()
        for filename in ("up.csv", "down.csv"):
            with (legacy / filename).open("w", newline="") as f:
                csv.writer(f).writerows([["id"]+["v"]*20, [names[0]]+[""]*20, [names[1]]+[".1"]*20])
        with (legacy / "area.csv").open("w", newline="") as f:
            csv.writer(f).writerows([["id"]+["v"]*10+["label"], ["2026.1.1.0_2026.1.1.1"]+[".2"]*10+["1"]])
        self.config = dict(host="127.0.0.1", port=8877, device="cpu", threads=1,
                           output=str(self.root / "output"), dataset=str(self.root / "dataset"),
                           legacy_features=str(legacy), legacy_checkpoint=str(self.root / "missing.pth"),
                           profiles={mode:dict(name=mode, checkpoint="absent.pt", classifier=None) for mode in ("day", "night")},
                           quality=dict(blur_min=0, dark_max=1, bright_max=1), alert=dict(ALERT))
        self.monitor = Monitor(copy.deepcopy(self.config))

    def tearDown(self):
        self.monitor.pool.shutdown(wait=True)
        self.tmp.cleanup()

    def test_exact_frame_identity_required_for_legacy_features(self):
        pairs = self.monitor.catalog.pairs
        self.assertEqual(len(pairs), 2)
        self.assertTrue(pairs[0]["legacy_available"])
        self.assertFalse(pairs[1]["legacy_available"])
        self.assertEqual(self.monitor.catalog.legacy[pairs[0]["id"]][0].shape, (2,20))

    def test_duplicate_legacy_rows_are_excluded(self):
        folder = self.root / "legacy"
        for filename in ("up.csv", "down.csv", "area.csv"):
            with (folder / filename).open(newline="") as f:
                rows = list(csv.reader(f))
            with (folder / filename).open("w", newline="") as f:
                csv.writer(f).writerows(rows + rows[1:])
        self.assertFalse(Catalog(self.config).legacy)

    def test_image_classifier_checks_upstream_and_uses_frozen_threshold(self):
        from riverlab.features import pair_geometry
        from riverlab.classification import feature_columns
        mask = np.zeros((64,64), dtype=np.uint8)
        mask[20:40,10:50] = 1
        image = Image.new("RGB", (64,64), (100,110,120))
        rows = [pair_geometry([mask,mask],[image,image])]
        payload = dict(kind="area_threshold", columns=feature_columns(rows,"area"),
                       config=dict(quality_gate=False), threshold=.12,
                       upstream=dict(source="predicted",feature_kind="pair-features",checkpoint_sha256="correct"))
        path = self.root / "classifier.joblib"
        joblib.dump(payload,path)
        self.monitor.config["profiles"]["day"]["classifier"] = str(path)
        with self.assertRaisesRegex(ValueError,"不匹配"):
            self.monitor.engine.image_classifier("day",[mask,mask],[image,image],"wrong")
        score, sha, threshold = self.monitor.engine.image_classifier("day",[mask,mask],[image,image],"correct")
        self.assertEqual(score,0)
        self.assertEqual(threshold,.12)
        self.assertEqual(len(sha),64)

    def fake_engine(self, pair, folder):
        folder.mkdir(parents=True)
        return dict(pair, prediction=1, score=.9, threshold=.7, source="historical_features")

    def await_job(self, job_id):
        for _ in range(200):
            job = self.monitor.job(job_id)
            if job["status"] not in {"queued", "running"}:
                return job
            time.sleep(.01)
        self.fail("Job did not finish")

    def test_independent_pairs_never_form_confirmed_event_and_history_persists(self):
        self.monitor.engine.analyze = self.fake_engine
        result = self.monitor.start_catalog({"pair_ids":[p["id"] for p in self.monitor.catalog.pairs]})
        job = self.await_job(result["job_id"])
        self.assertEqual(job["status"], "complete")
        self.assertEqual([r["alarm"] for r in job["results"]], ["candidate", "candidate"])
        self.assertEqual(len(Store(self.root / "output").history()), 2)

    def test_single_worker_rejects_second_task_and_stops_after_inflight_image(self):
        entered, release = threading.Event(), threading.Event()
        def slow(pair, folder):
            entered.set()
            release.wait(3)
            return self.fake_engine(pair, folder)
        self.monitor.engine.analyze = slow
        ids = [p["id"] for p in self.monitor.catalog.pairs]
        job_id = self.monitor.start_catalog({"pair_ids":ids})["job_id"]
        self.assertTrue(entered.wait(1))
        with self.assertRaises(ValueError):
            self.monitor.start_catalog({"pair_ids":ids})
        self.monitor.job(job_id, True)
        release.set()
        job = self.await_job(job_id)
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(job["done"], 1)

    def test_uploads_do_not_borrow_legacy_features_from_filename(self):
        buf = io.BytesIO()
        Image.new("RGB", (64,64)).save(buf, "PNG")
        data = base64.b64encode(buf.getvalue()).decode()
        self.monitor.engine.analyze = lambda pair, folder: dict(pair, prediction=None, score=None, threshold=.7, source="unavailable")
        files = [dict(name="202601010{}0000.jpg".format(i), timestamp="2026-01-01T0{}:00:00".format(i), data=data) for i in range(3)]
        job_id = self.monitor.upload(dict(mode="day", files=files))["job_id"]
        job = self.await_job(job_id)
        self.assertEqual(job["kind"], "sequence")
        self.assertEqual([r["alarm"] for r in job["results"]], ["unknown", "unknown"])
        self.assertTrue(all(not r["legacy_available"] for r in job["results"]))
        self.assertEqual(job["results"][0]["reference"], job["results"][1]["reference"])
        restarted = Monitor(copy.deepcopy(self.config))
        try:
            self.assertIn(job["results"][0]["reference"], restarted.catalog.files)
            self.assertEqual(len(restarted.store.history()), 2)
        finally:
            restarted.pool.shutdown(wait=True)
        files[1]["timestamp"] = files[0]["timestamp"]
        with self.assertRaises(ValueError):
            self.monitor.upload(dict(mode="day", files=files))

    def test_failed_observation_breaks_confirmation(self):
        pairs = []
        for i in range(4):
            pairs.append(dict(self.monitor.catalog.pairs[0], id=str(i), timestamp="2026-01-01T00:0{}:00".format(i)))
        def analyze(pair, folder):
            if pair["id"] == "1":
                raise ValueError("unreadable frame")
            return self.fake_engine(pair, folder)
        self.monitor.engine.analyze = analyze
        job = self.await_job(self.monitor.start(pairs,"sequence")["job_id"])
        self.assertEqual(job["status"],"partial")
        self.assertEqual([r["alarm"] for r in job["results"]],["pending","pending","triggered"])

    def test_http_origin_token_and_asset_limits(self):
        server = ThreadingHTTPServer(("127.0.0.1",0), handler_for(self.monitor))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = "http://127.0.0.1:"+str(server.server_port)
        try:
            with urllib.request.urlopen(url+"/api/bootstrap") as response:
                self.assertEqual(len(json.load(response)["pairs"]), 2)
            request = urllib.request.Request(url+"/api/settings", data=json.dumps(ALERT).encode(), headers={"Content-Type":"application/json"})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, 403)
            request.add_header("X-Monitor-Token", self.monitor.token)
            request.add_header("Origin", "https://other.example")
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, 403)
            for path in ("/results/../../monitor.sqlite3", "/api/image/unknown", "/../configs/monitor.json"):
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(url+path)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
