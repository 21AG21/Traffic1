# 02_collect_events.py
import csv, os, sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

URL = "https://api.511.org/traffic/events"
OUT = Path("data/fremont_events_log.csv")
FIELDS = ["pulled_at", "id", "event_type", "severity", "headline", "roads", "created", "updated"]


def fetch_events(token):
    r = requests.get(URL, params={"api_key": token, "format": "json"}, timeout=30)
    r.raise_for_status()
    return r.json() if not r.content.startswith(b"\xef\xbb\xbf") else __import__("json").loads(r.content.decode("utf-8-sig"))


def flatten(event, pulled_at):
    roads = event.get("roads") or []
    return {
        "pulled_at": pulled_at,
        "id": event.get("id"),
        "event_type": event.get("event_type"),
        "severity": event.get("severity"),
        "headline": (event.get("headline") or "").replace("\n", " "),
        "roads": "; ".join(r.get("name", "") for r in roads),
        "created": event.get("created"),
        "updated": event.get("updated"),
    }


def main():
    load_dotenv()
    token = os.environ.get("TOKEN_511")
    if not token:
        sys.exit("No TOKEN_511 in .env — see step 10.")

    try:
        payload = fetch_events(token)
    except requests.RequestException as e:
        sys.exit(f"Could not reach 511: {e}")

    events = payload.get("events", [])
    print(f"{len(events)} events came back")
    if events:
        print("fields on the first one:", sorted(events[0].keys()))

    pulled_at = datetime.now(timezone.utc).isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUT.exists()
    with OUT.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for e in events:
            w.writerow(flatten(e, pulled_at))
    print(f"wrote {len(events)} rows to {OUT}")


if __name__ == "__main__":
    main()