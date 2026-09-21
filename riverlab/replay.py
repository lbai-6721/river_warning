"""Causal alert replay with reviewed event intervals and explicit monitoring time."""
from collections import defaultdict
from datetime import datetime

import numpy as np

from .io import completed, new_run, read_csv, write_csv, write_json


def dt(value):
    return datetime.fromisoformat(value)


def replay(rows, events, monitoring, consecutive=1, max_gap_hours=2, warning_hours=0):
    if consecutive < 1 or max_gap_hours <= 0 or warning_hours < 0:
        raise ValueError("Invalid replay settings")
    if not monitoring:
        raise ValueError("Explicit monitoring intervals required for false-alarm rates")
    if len({(e["camera_id"], e["event_id"]) for e in events}) != len(events):
        raise ValueError("Duplicate event IDs within a camera")
    thresholds = {float(r["threshold"]) for r in rows}
    if len(thresholds) != 1 or not all(np.isfinite(float(r["score"])) for r in rows):
        raise ValueError("Replay needs finite scores and one frozen threshold")
    keys = [(r["camera_id"], r["timestamp"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Replay requires one prediction per camera/time; arbitrary image pairs are not a stream")
    for m in monitoring:
        if m.get("reviewed") != "1" or m["state"] not in {"normal", "event", "unknown"}:
            raise ValueError("Monitoring intervals need reviewed=1 and a state")
        if dt(m["start"]) >= dt(m["end"]):
            raise ValueError("Monitoring interval duration must be positive")
    for a in monitoring:
        for b in monitoring:
            if a is b or a["camera_id"] != b["camera_id"]:
                continue
            if max(dt(a["start"]), dt(b["start"])) < min(dt(a["end"]), dt(b["end"])):
                raise ValueError("Overlapping monitoring intervals")
    for e in events:
        if e.get("reviewed") != "1" or not e.get("event_id") or dt(e["end"]) < dt(e["start"]):
            raise ValueError("Reviewed events with valid onset/end required")
        intervals = [m for m in monitoring if m["camera_id"] == e["camera_id"]]
        if not any(m["state"] == "event" and dt(m["start"]) <= dt(e["start"]) and
                   dt(m["end"]) >= dt(e["end"]) for m in intervals):
            raise ValueError("Each event must be contained in a reviewed event monitoring interval")
    def interval_for(r):
        return next((m for m in monitoring if m["camera_id"] == r["camera_id"]
                     and dt(m["start"]) <= dt(r["timestamp"]) < dt(m["end"])), None)
    streams = defaultdict(list)
    for r in rows:
        if interval_for(r) is None:
            raise ValueError("Prediction outside monitored intervals")
        streams[r["camera_id"]].append(r)
    alerts, normal_hours = [], 0.0
    for camera, stream in streams.items():
        stream.sort(key=lambda r: r["timestamp"])
        streak, active, last = 0, False, None
        for row in stream:
            t = dt(row["timestamp"])
            gap = (t-dt(last["timestamp"])).total_seconds()/3600 if last else None
            if last and 0 < gap <= max_gap_hours:
                if str(last.get("valid", 1)) == "1" and str(row.get("valid", 1)) == "1":
                    for m in monitoring:
                        if m["camera_id"] == camera and m["state"] == "normal":
                            overlap = (min(t, dt(m["end"])) -
                                       max(dt(last["timestamp"]), dt(m["start"]))).total_seconds()
                            normal_hours += max(0, overlap)/3600
            if last is None or gap > max_gap_hours or gap <= 0:
                streak, active = 0, False
            positive = str(row.get("valid", 1)) == "1" and float(row["score"]) >= float(row["threshold"])
            streak = streak+1 if positive else 0
            if not positive:
                active = False
            if streak >= consecutive and not active:
                alerts.append({"camera_id": camera, "timestamp": row["timestamp"],
                               "score": float(row["score"]), "monitoring_state": interval_for(row)["state"]})
                active = True
            last = row
    matched = []
    used_alerts = set()
    for e in sorted(events, key=lambda e: e["start"]):
        candidates = []
        for i, a in enumerate(alerts):
            lead = (dt(e["start"])-dt(a["timestamp"])).total_seconds()/3600
            if i not in used_alerts and a["camera_id"] == e["camera_id"] and (
                    dt(a["timestamp"]) <= dt(e["end"]) and lead <= warning_hours):
                candidates.append((dt(a["timestamp"]), i))
        match = min(candidates)[1] if candidates else None
        if match is not None:
            used_alerts.add(match)
        observed = [r for r in rows if r["camera_id"] == e["camera_id"] and
                    dt(e["start"]) <= dt(r["timestamp"]) <= dt(e["end"])]
        matched.append({
            "event_id": e["event_id"], "camera_id": e["camera_id"], "start": e["start"],
            "detected": int(match is not None),
            "alert_time": alerts[match]["timestamp"] if match is not None else "",
            "delay_hours": ((dt(alerts[match]["timestamp"])-dt(e["start"])).total_seconds()/3600
                            if match is not None else ""),
            "observations": len(observed),
            "valid_observations": sum(str(r.get("valid", 1)) == "1" for r in observed)})
    false = sum(a["monitoring_state"] == "normal" and i not in used_alerts for i, a in enumerate(alerts))
    delays = [float(m["delay_hours"]) for m in matched if m["detected"]]
    total = sum((dt(m["end"])-dt(m["start"])).total_seconds()/3600 for m in monitoring)
    metrics = {"events": len(events), "detected_events": len(delays),
               "total_alert_episodes": len(alerts),
               "unmatched_alerts_in_event_intervals": sum(
                   a["monitoring_state"] == "event" and i not in used_alerts for i, a in enumerate(alerts)),
               "event_recall": len(delays)/len(events) if events else None,
               "false_alerts": false, "valid_normal_hours": normal_hours,
               "false_alerts_per_day": false*24/normal_hours if normal_hours else None,
               "delay_hours_median_detected_only": float(np.median(delays)) if delays else None,
               "monitored_hours_declared": total,
               "valid_frame_fraction": np.mean([str(r.get("valid", 1)) == "1" for r in rows]) if rows else None,
               "events_without_valid_observation": sum(m["valid_observations"] == 0 for m in matched),
               "unmatched_alerts_in_unknown_intervals": sum(a["monitoring_state"] == "unknown" for a in alerts),
               "warning_horizon_hours": warning_hours,
               "normal_exposure_definition": "overlap of normal intervals with consecutive valid observations no farther apart than max_gap_hours"}
    return metrics, alerts, matched


def run(predictions, events, monitoring, output, **options):
    rows, ev, intervals = read_csv(predictions), read_csv(events), read_csv(monitoring)
    result, alerts, matches = replay(rows, ev, intervals, **options)
    directory = new_run(output, options, [predictions, events, monitoring])
    write_json(directory / "metrics.json", result)
    write_csv(directory / "alerts.csv", alerts,
              ["camera_id", "timestamp", "score", "monitoring_state"])
    write_csv(directory / "events.csv", matches,
              ["event_id", "camera_id", "start", "detected", "alert_time", "delay_hours",
               "observations", "valid_observations"])
    from .reporting import timeline
    timeline(rows, ev, directory / "timeline.png")
    completed(directory)
    return result
