import unittest
from riverlab.replay import replay


def t(hour):
    return "2022-06-01T{:02d}:00:00".format(hour)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(camera_id="day", timestamp=t(i), valid="1",
                          score=.9 if i in [2, 6, 7] else .1, threshold=.5) for i in range(10)]
        self.events = [dict(camera_id="day", start=t(6), end=t(8), event_id="e", reviewed="1")]
        self.monitor = [dict(camera_id="day", start=t(0), end=t(6), state="normal", reviewed="1"),
                        dict(camera_id="day", start=t(6), end=t(10), state="event", reviewed="1")]

    def test_event_detection_false_alert_and_exposure(self):
        result, alerts, events = replay(self.rows, self.events, self.monitor)
        self.assertEqual(result["event_recall"], 1)
        self.assertEqual(result["false_alerts"], 1)
        self.assertEqual(result["valid_normal_hours"], 6)
        self.assertEqual(result["false_alerts_per_day"], 4)
        self.assertEqual(events[0]["delay_hours"], 0)

    def test_consecutive_rule_uses_only_past_frames(self):
        result, alerts, events = replay(self.rows, self.events, self.monitor, consecutive=2)
        self.assertEqual(result["false_alerts"], 0)
        self.assertEqual(events[0]["delay_hours"], 1)

    def test_rejected_event_still_counts_as_missed(self):
        for row in self.rows[6:9]:
            row["valid"] = "0"
        result, _, _ = replay(self.rows, self.events, self.monitor)
        self.assertEqual(result["event_recall"], 0)
        self.assertEqual(result["events_without_valid_observation"], 1)

    def test_arbitrary_duplicate_pairs_cannot_be_replayed(self):
        with self.assertRaisesRegex(ValueError, "one prediction"):
            replay(self.rows + [self.rows[0]], self.events, self.monitor)
