"""Poll the 511.org Open511 events feed and keep the Fremont events.

Run every 15 minutes (launchd on the Mac mini, not cron). Each poll's raw
response is appended to data/events_raw/YYYY-MM-DD.jsonl, so the filter rules
can change later without polling again. The deduplicated table is
data/fremont_events.csv, one row per event.

Usage:
    python pipeline/collect_511.py            # one poll (what launchd runs)
"""
import argparse
import csv
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from config import DATA, ENV_FILE, FREMONT_ROAD_RE, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN

API_URL = "https://api.511.org/traffic/events"
EVENTS_CSV = DATA / "fremont_events.csv"
RAW_DIR = DATA / "events_raw"

FIELDS = [
    "event_id", "event_type", "severity", "status", "headline", "created",
    "updated", "road_names", "directions", "lat", "lon", "geo_missing",
    "first_seen_at", "last_seen_at",
]


def load_env(path=ENV_FILE):
    """Read KEY=VALUE lines from .env into a dict. Ignores blanks and comments."""
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def fetch_events(api_key, status="active"):
    """Return the list of events from one API call."""
    query = urllib.parse.urlencode({"api_key": api_key, "format": "json", "status": status})
    with urllib.request.urlopen(f"{API_URL}?{query}", timeout=30) as response:
        body = json.loads(response.read().decode("utf-8-sig"))
    return body.get("events", [])


def point(event):
    """Return (lon, lat) from a GeoJSON-style geography, or (None, None)."""
    coords = (event.get("geography") or {}).get("coordinates")
    if not coords or len(coords) != 2:
        return None, None
    return coords[0], coords[1]


def is_fremont(event):
    """Keep an event on a Fremont freeway name, or with a point inside the box."""
    names = " ".join(road.get("name", "") for road in event.get("roads", []))
    if FREMONT_ROAD_RE.search(names):
        return True
    lon, lat = point(event)
    return lat is not None and LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


def flatten(event):
    """One flat row per event, with the nested road list joined into text."""
    lon, lat = point(event)
    roads = event.get("roads", [])
    return {
        "event_id": event["id"],
        "event_type": event.get("event_type", ""),
        "severity": event.get("severity", ""),
        "status": event.get("status", ""),
        "headline": event.get("headline", ""),
        "created": event.get("created", ""),
        "updated": event.get("updated", ""),
        "road_names": "; ".join(road.get("name", "") for road in roads),
        "directions": "; ".join(road.get("direction", "") for road in roads),
        "lat": "" if lat is None else lat,
        "lon": "" if lon is None else lon,
        "geo_missing": int(lat is None),
    }


def merge_into_log(rows, pulled_at, path=EVENTS_CSV):
    """Upsert rows by event_id. Keep the first_seen_at of older events."""
    existing = {}
    if path.exists():
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                existing[row["event_id"]] = row

    for row in rows:
        previous = existing.get(row["event_id"])
        row["first_seen_at"] = previous["first_seen_at"] if previous else pulled_at
        row["last_seen_at"] = pulled_at
        existing[row["event_id"]] = row

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing.values())
    return len(existing)


def append_raw(events, pulled_at, raw_dir=RAW_DIR):
    """Append the whole poll to today's raw JSONL file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    day = pulled_at[:10]
    with (raw_dir / f"{day}.jsonl").open("a") as f:
        f.write(json.dumps({"pulled_at": pulled_at, "events": events}) + "\n")


def poll_once(api_key, status="active"):
    pulled_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    events = fetch_events(api_key, status)
    append_raw(events, pulled_at)
    fremont = [flatten(event) for event in events if is_fremont(event)]
    total = merge_into_log(fremont, pulled_at)
    print(f"{pulled_at}: {len(events)} events from 511, {len(fremont)} in Fremont, {total} in log")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", default="active", choices=["active", "archived"])
    args = parser.parse_args()

    api_key = load_env().get("TOKEN_511")
    if not api_key:
        raise SystemExit("TOKEN_511 is missing from .env")
    poll_once(api_key, args.status)


if __name__ == "__main__":
    main()
