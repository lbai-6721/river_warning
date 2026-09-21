import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from riverlab.augment import apply
from riverlab.cli import main
from riverlab.features import audit_legacy_features, load_legacy, extract_pair_features
from riverlab.io import write_csv, write_json, read_json, new_run, completed
from riverlab.replay import run as replay_run
from riverlab.reporting import classifier_figures, aggregate
from riverlab.suite import plan
from riverlab.data import scan_dataset, audit_masks
from riverlab.masks import decode_mask


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="river_artifact_")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_legacy_alignment_diagnostics_preserve_source(self):
        a = "cam_20220601100000_TIMING.jpg"
        b = "cam_20220601110000_TIMING.jpg"
        for name in ["up.csv", "down.csv"]:
            with (self.root/name).open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows([["id"]+["v"]*20, [a]+[""]*20, [b]+["0.1"]*20])
        area = self.root/"area.csv"
        with area.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([["id"]+["v"]*10+["label"],
                                    ["2022.6.1.10_2022.6.1.11"]+["0.2"]*10+["1"]])
        self.assertTrue(audit_legacy_features(self.root)["valid"])
        self.assertEqual(len(load_legacy(self.root)), 1)
        area.write_text(area.read_text().replace("2022.6.1.10_", "2022.6.1.11_"), encoding="utf-8")
        original = area.read_bytes()
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            main(["import-legacy", "--folder", str(self.root), "--output", str(self.root/"out.csv")])
        self.assertEqual(area.read_bytes(), original)
        self.assertEqual(read_json(self.root/"out.audit.json")["issue_pairs"], 1)
        self.assertFalse((self.root/"out.csv").exists())

    def test_weather_and_device_augmentation_preserve_labels(self):
        image = np.full((32, 48, 3), 80, dtype=np.uint8)
        mask = np.zeros((32, 48), np.uint8)
        mask[:, 10:20] = 1
        for seed in range(12):
            a, b = apply(image, mask, np.random.default_rng(seed),
                         {"weather_p": 1, "equipment_p": 1})
            self.assertEqual(a.shape, image.shape)
            np.testing.assert_array_equal(mask, b)
        with self.assertRaises(ValueError):
            apply(image, mask, np.random.default_rng(1), {"weather_p": 2})

    def test_planning_is_dry_and_expands_seeds(self):
        config = self.root/"base.json"
        write_json(config, {"model": {"name": "unet"}})
        result = plan({"experiments": [{"name": "example", "command": "train-seg",
                        "base_config": str(config), "grid": {"seed": [42, 2026], "model.name": ["unet", "segnet"]}}]},
                      self.root/"plan")
        self.assertEqual(result["jobs"], 4)
        self.assertEqual(read_json(self.root/"plan"/"example_002.json")["seed"], 2026)
        self.assertEqual(len(list((self.root/"plan").glob("*.pt"))), 0)

    def test_plot_and_aggregate_exports_real_predictions(self):
        predictions = self.root/"predictions.csv"
        write_csv(predictions, [dict(y_true=y, y_pred=int(s >= .5), score=s, valid=1)
                               for y, s in [(0, .1), (1, .9), (0, .6), (1, .7)]])
        classifier_figures(predictions, self.root/"figures")
        for name in ["confusion_matrix.png", "precision_recall.png"]:
            with Image.open(self.root/"figures"/name) as im:
                self.assertGreater(im.width, 100)
        run = new_run(self.root/"run", {"seed": 42}, [predictions])
        write_json(run/"metrics.json", {"f1": .8})
        completed(run)
        table = aggregate([run], self.root/"results.csv")
        self.assertEqual(table[0]["f1"], .8)
        self.assertEqual(read_json(run/"run.json")["status"], "complete")
        with self.assertRaises(FileExistsError):
            new_run(run, {})

    def test_replay_exports_even_when_no_alerts(self):
        predictions, events, monitor = [self.root/name for name in ["pred.csv", "events.csv", "monitor.csv"]]
        write_csv(predictions, [dict(camera_id="day", timestamp="2022-06-01T{:02d}:00:00".format(i),
                                    score=.1, threshold=.5, valid=1) for i in range(3)])
        write_csv(events, [], ["event_id", "camera_id", "start", "end", "reviewed"])
        write_csv(monitor, [dict(camera_id="day", start="2022-06-01T00:00:00",
                                 end="2022-06-01T03:00:00", state="normal", reviewed=1)])
        result = replay_run(predictions, events, monitor, self.root/"replay")
        self.assertEqual(result["false_alerts"], 0)
        self.assertIsNone(result["event_recall"])
        with Image.open(self.root/"replay"/"timeline.png") as im:
            self.assertGreater(im.width, 100)

    def test_json_accepts_windows_utf8_bom(self):
        path = self.root/"config.json"
        path.write_text('{"seed":42}', encoding="utf-8-sig")
        self.assertEqual(read_json(path)["seed"], 42)

    def test_inventory_fast_path_and_explicit_size_validation(self):
        for name in ["JPEGImages", "SegmentationClass"]:
            (self.root/name).mkdir()
        stem = "172.16.29.10_01_20220601120000_TIMING_00"
        image, mask = self.root/"JPEGImages"/(stem+".jpg"), self.root/"SegmentationClass"/(stem+".png")
        Image.fromarray(np.zeros((16, 24, 3), np.uint8)).save(image)
        Image.fromarray(np.zeros((16, 24), np.uint8)).save(mask)
        rows = scan_dataset(self.root, "day", "day")
        self.assertEqual(rows[0]["width"], "")
        self.assertEqual(rows[0]["parent_id"], "day:172.16.29.10_01_20220601120000_TIMING")
        self.assertEqual(scan_dataset(self.root, "day", "day", check_sizes=True)[0]["width"], 24)
        audit_masks(rows)
        Image.fromarray(np.zeros((8, 24), np.uint8)).save(mask)
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            audit_masks(rows)

    def test_explicit_rgb_label_decoding_and_unknown_rejection(self):
        mapping = read_json("configs/day_mask_mapping.json")
        rgb = np.array([[[0, 0, 0], [128, 0, 0], [1, 1, 1]]], dtype=np.uint8)
        np.testing.assert_array_equal(decode_mask(rgb, mapping), [[0, 1, 1]])
        with self.assertRaisesRegex(ValueError, "explicit"):
            decode_mask(rgb)
        rgb[0, 2] = [127, 0, 0]
        with self.assertRaisesRegex(ValueError, "Unmapped"):
            decode_mask(rgb, mapping)
