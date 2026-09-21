import unittest
from riverlab.data import (timestamp, parent_id, group_rows, split_rows,
                           validate_splits, ensure_unseen, provenance)


def row(i, day=None, **extra):
    r = dict(sample_id="s"+str(i), parent_id="p"+str(i), camera_id="day",
             timestamp="2022-06-{:02d}T12:00:00".format(day or i+1), label=str(i % 2))
    r.update(extra)
    return r


class DataTests(unittest.TestCase):
    def test_dotted_camera_and_extensionless_parent(self):
        name = "172.16.29.10_01_20220601094610600_TIMING_41"
        expected = "172.16.29.10_01_20220601094610600_TIMING"
        self.assertEqual(parent_id(name), expected)
        self.assertEqual(parent_id(name+".jpg"), expected)
        self.assertEqual(timestamp(name), "2022-06-01T09:46:10")
        with self.assertRaises(ValueError):
            timestamp("frame_001.jpg")

    def test_both_cameras_same_date_stay_together(self):
        rows = group_rows([row(0), row(1, day=1, camera_id="night"), row(2)])
        self.assertEqual(rows[0]["group_id"], rows[1]["group_id"])
        self.assertNotEqual(rows[1]["group_id"], rows[2]["group_id"])

    def test_shared_frames_transitive(self):
        rows = group_rows([row(0, frame_a="a", frame_b="b"),
                           row(1, frame_a="b", frame_b="c"), row(2, frame_a="c", frame_b="d")])
        self.assertEqual(len({r["group_id"] for r in rows}), 1)

    def test_fixed_test_in_group_cv(self):
        rows, folds = split_rows([row(i) for i in range(15)], folds=3)
        held = {r["sample_id"] for r in rows if r["split"] == "test"}
        self.assertEqual(len(folds), 3)
        for fold in folds:
            validate_splits(fold)
            self.assertEqual(held, {r["sample_id"] for r in fold if r["split"] == "test"})

    def test_relabelled_groups_cannot_hide_shared_frame(self):
        rows = [row(0, split="train", group_id="a"),
                row(1, parent_id="p0", split="test", group_id="b")]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_splits(rows, require_all=False)

    def test_model_provenance_checks_dates_across_manifests(self):
        train = [row(0, split="train", group_id="a"), row(1, split="val", group_id="b")]
        test = [row(8, day=1, split="test", group_id="new")]
        with self.assertRaisesRegex(ValueError, "development"):
            ensure_unseen(test, provenance(train))

    def test_fewer_than_three_groups_refused(self):
        with self.assertRaises(ValueError):
            split_rows([row(0), row(1)])

    def test_event_groups_stay_together(self):
        values = group_rows([row(0, event_id="event"), row(1, event_id="event"), row(2)])
        self.assertEqual(values[0]["group_id"], values[1]["group_id"])
