"""Tests for collect_511.py, using two hand-written events."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from collect_511 import flatten, is_fremont, merge_into_log, FIELDS  # noqa: E402

FREMONT_EVENT = {
    "id": "EV-1",
    "event_type": "INCIDENT",
    "severity": "Moderate",
    "status": "ACTIVE",
    "headline": "Collision on I-880 NB",
    "created": "2026-01-15T08:02:00-08:00",
    "updated": "2026-01-15T08:20:00-08:00",
    "roads": [{"name": "I-880", "direction": "NORTH"}, {"name": "Fremont Blvd", "direction": "NORTH"}],
    "geography": {"type": "Point", "coordinates": [-121.97, 37.53]},
}

ELSEWHERE_EVENT = {
    "id": "EV-2",
    "event_type": "ROADWORK",
    "severity": "Minor",
    "status": "ACTIVE",
    "headline": "Lane closure on US-101 in San Jose",
    "created": "2026-01-15T09:00:00-08:00",
    "updated": "2026-01-15T09:00:00-08:00",
    "roads": [{"name": "US-101", "direction": "SOUTH"}],
    "geography": {"type": "Point", "coordinates": [-121.89, 37.33]},
}


def test_fremont_road_name_keeps_event():
    assert is_fremont(FREMONT_EVENT)


def test_event_outside_box_and_road_is_dropped():
    assert not is_fremont(ELSEWHERE_EVENT)


def test_event_without_geography_kept_by_road_name():
    event = dict(FREMONT_EVENT, geography=None)
    assert is_fremont(event)
    row = flatten(event)
    assert row["geo_missing"] == 1
    assert row["lat"] == ""


def test_flatten_joins_roads_into_text():
    row = flatten(FREMONT_EVENT)
    assert row["road_names"] == "I-880; Fremont Blvd"
    assert row["directions"] == "NORTH; NORTH"
    assert row["geo_missing"] == 0
    assert row["lon"] == -121.97


def test_merge_dedupes_and_keeps_first_seen(tmp_path):
    log = tmp_path / "events.csv"
    first = flatten(FREMONT_EVENT)
    merge_into_log([first], "2026-01-15T16:00:00Z", log)

    second = flatten(dict(FREMONT_EVENT, updated="2026-01-15T08:45:00-08:00"))
    total = merge_into_log([second], "2026-01-15T16:15:00Z", log)

    assert total == 1
    with log.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == FIELDS
    assert rows[0]["first_seen_at"] == "2026-01-15T16:00:00Z"
    assert rows[0]["last_seen_at"] == "2026-01-15T16:15:00Z"
    assert rows[0]["updated"] == "2026-01-15T08:45:00-08:00"
