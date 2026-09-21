import unittest
import numpy as np
from riverlab.features import geometry, boundary_motion, temporal_features
from riverlab.metrics import classification, confusion_seg, segmentation, boundary_counts, boundary_f1
from riverlab.classification import TrainOnlyScaler, matrix


class GeometryTests(unittest.TestCase):
    def test_last_strip_and_resolution_normalized_area(self):
        mask = np.zeros((20, 40), np.uint8)
        mask[5:15, :] = 1
        g = geometry(mask)
        self.assertEqual(len([k for k in g if k.startswith("area_")]), 20)
        self.assertAlmostEqual(g["area_19"], .5)
        larger = np.repeat(np.repeat(mask, 2, 0), 2, 1)
        self.assertAlmostEqual(geometry(larger)["area_19"], g["area_19"])

    def test_missing_boundary_is_not_zero(self):
        g = geometry(np.zeros((20, 20), np.uint8))
        self.assertTrue(np.isnan(g["upper_00"]))
        self.assertEqual(g["coverage_00"], 0)

    def test_absolute_motion_does_not_cancel(self):
        a = np.zeros((20, 20), np.uint8)
        a[7:12, :] = 1
        b = np.zeros_like(a)
        b[5:10, :10], b[9:14, 10:] = 1, 1
        motion = boundary_motion(a, b, bins=1)
        self.assertAlmostEqual(motion["upper_00"], 0)
        self.assertAlmostEqual(motion["upper_abs_00"], .1)
        smooth = boundary_motion(a, b, bins=20, sigma=2)
        self.assertLess(smooth["upper_abs_09"], .1)

    def test_temporal_windows_are_causal_and_gap_limited(self):
        rows = [dict(sample_id=str(i), parent_id=str(i), camera_id="day", group_id="g",
                     split="test", mode="day", label="1", quality_valid=1,
                     timestamp="2022-06-01T{:02d}:00:00".format(t), area_00=i)
                for i, t in enumerate([0, 1, 2, 12])]
        result = temporal_features(rows, lags=[1, 2], max_gap_hours=3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["sample_id"], "2")
        self.assertEqual(result[0]["area_00_lag2"], 2)
        rows[1]["split"] = "train"
        self.assertEqual(temporal_features(rows, lags=[1, 2]), [])

    def test_scaler_uses_only_training_and_keeps_missing_columns(self):
        x = np.array([[0, np.nan], [2, np.nan]])
        scaler = TrainOnlyScaler().fit(x)
        self.assertEqual(scaler.transform(np.array([[100, np.nan]])).tolist(), [[99., 0.]])
        self.assertEqual(matrix([{"area_00": 0}], ["area_00"])[0, 0], 0)


class MetricsTests(unittest.TestCase):
    def test_whole_sample_metrics(self):
        result = classification([1, 0, 1], [.9, .8, .7], .5)
        self.assertAlmostEqual(result["accuracy"], 2/3)
        self.assertEqual(result["confusion_matrix"], [[0, 1], [0, 2]])
        self.assertIsNone(classification([1], [.9])["average_precision"])

    def test_segmentation_void_ignored(self):
        truth = np.array([[0, 1], [1, 255]])
        prediction = np.array([[0, 1], [0, 1]])
        result = segmentation(confusion_seg(truth, prediction))
        self.assertEqual(result["river_iou"], .5)
        self.assertAlmostEqual(result["dice"], 2/3)

    def test_boundary_metrics_measure_alignment(self):
        a = np.zeros((40, 40), np.uint8)
        a[8:25, 8:25] = 1
        self.assertEqual(boundary_f1(boundary_counts(a, a, 1)), 1)
        self.assertLess(boundary_f1(boundary_counts(a, np.roll(a, 7, 0), 1)), 1)
