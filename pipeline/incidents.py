"""Stage 7: join 511 incidents onto every station and 5-minute step.

A station is affected by an event when the event is on the same freeway and
direction and lies within RADIUS_MILES of the station. An event with no
coordinates counts for every station on its freeway and direction in the
registry, and is flagged by its distance being set to RADIUS_MILES.

An event is active in the archive from `created` until the later of `updated`
and `created` plus 20 minutes. The live loop uses first_seen and last_seen
instead, so the two windows are not the same (see the plan, stage 7).

Output columns, aligned row-for-row with the grid:
  incident_active, incident_severity (0 none, 1 minor or unknown, 2 moderate, 3 major),
  minutes_since_incident_start (0 when none), nearest_incident_miles (99 when none).
"""
import re

import numpy as np
import pandas as pd

ROUTE_RE = re.compile(r"(?:I-?|SR-?)(880|680|84)\b", re.IGNORECASE)
SEVERITY_CODE = {"minor": 1, "moderate": 2, "major": 3}
MIN_ACTIVE = pd.Timedelta(minutes=20)
RADIUS_MILES = 1.0
NO_INCIDENT_DISTANCE = 99.0


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles between two points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def prepare_events(events):
    """Turn the events CSV into one row per event with route, direction and active window."""
    out = pd.DataFrame({"event_id": events["event_id"].astype(str)})
    routes = events["road_names"].fillna("").str.extract(ROUTE_RE, flags=re.IGNORECASE)[0]
    out["freeway"] = routes.astype("string")
    out["direction"] = events["directions"].fillna("").str.split(";").str[0].str.strip().str[:1].str.upper()
    out["start"] = pd.to_datetime(events["created"], utc=True)
    updated = pd.to_datetime(events["updated"], utc=True)
    out["end"] = np.maximum(updated, out["start"] + MIN_ACTIVE)
    out["severity"] = events["severity"].fillna("").str.lower().map(SEVERITY_CODE).fillna(1).astype(int)
    out["lat"] = pd.to_numeric(events["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(events["lon"], errors="coerce")
    out["geo_missing"] = events["geo_missing"].astype(int)
    return out.dropna(subset=["freeway", "start", "end"])


def join_incidents(grid, registry, events):
    """Return a DataFrame of the four incident columns, aligned to grid rows.

    grid must be sorted by station_id and timestamp, with a unique index.
    """
    freeway = registry.set_index("station_id")
    n = len(grid)
    active = np.zeros(n, dtype=np.int8)
    severity = np.zeros(n, dtype=np.int8)
    minutes = np.full(n, np.nan, dtype=np.float32)
    nearest = np.full(n, np.nan, dtype=np.float32)

    positions = grid.groupby("station_id", observed=True, sort=False).indices
    times = {sid: grid["timestamp"].to_numpy()[pos] for sid, pos in positions.items()}

    for event in prepare_events(events).itertuples(index=False):
        candidates = registry[(registry["freeway"].astype(str) == str(event.freeway))
                              & (registry["direction"] == event.direction)]
        if not event.geo_missing and not np.isnan(event.lat):
            dist = haversine_miles(event.lat, event.lon, candidates["lat"].to_numpy(), candidates["lon"].to_numpy())
            keep = dist <= RADIUS_MILES
            candidates = candidates.assign(distance=dist)[keep]
        else:
            candidates = candidates.assign(distance=RADIUS_MILES)

        start = np.datetime64(event.start.tz_convert("UTC").tz_localize(None), "ns")
        end = np.datetime64(event.end.tz_convert("UTC").tz_localize(None), "ns")
        for station_id, distance in zip(candidates["station_id"], candidates["distance"]):
            if station_id not in positions:
                continue
            ts = times[station_id]
            lo = np.searchsorted(ts, start, side="left")
            hi = np.searchsorted(ts, end, side="left")
            if lo >= hi:
                continue
            rows = positions[station_id][lo:hi]
            active[rows] = 1
            severity[rows] = np.maximum(severity[rows], event.severity)
            since = (ts[lo:hi] - start) / np.timedelta64(1, "m")
            minutes[rows] = np.fmin(minutes[rows], since.astype(np.float32))
            nearest[rows] = np.fmin(nearest[rows], np.float32(distance))

    return pd.DataFrame({
        "incident_active": active,
        "incident_severity": severity,
        "minutes_since_incident_start": np.nan_to_num(minutes, nan=0.0),
        "nearest_incident_miles": np.nan_to_num(nearest, nan=NO_INCIDENT_DISTANCE),
    }, index=grid.index)
