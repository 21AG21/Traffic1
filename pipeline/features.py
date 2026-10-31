"""Stage 8: feature table and forward-shifted targets.

Every feature at time t uses only data at or before t. The targets are the
label 15, 30 and 60 minutes after t, so the model forecasts instead of copying
the current state. Rows whose target is missing are dropped, never filled.

Neighbours are taken from the station registry: within each freeway and
direction, stations are sorted by postmile. Traffic comes from upstream, so
for travel direction N or E the upstream neighbour has the lower postmile,
and for S or W the higher. Check this against a known bottleneck before
trusting the neighbour features.

Built here: own speed, occupancy and flow history; rolling stats; the profile
ratio; two neighbour hops each way; incidents; calendar; static station fields.
Weather is joined when an hourly file is supplied. Not built yet: holiday
edge flags and the upstream flag on incidents.
"""
import numpy as np
import pandas as pd

STEPS_PER_MINUTE = 1 / 5
HORIZONS = {"y_15": 3, "y_30": 6, "y_60": 12}  # steps of 5 minutes
SPEED_LAGS = [1, 2, 3, 6, 12]
TARGET_COLUMNS = list(HORIZONS)
UPSTREAM_IF_LOWER_POSTMILE = {"N", "E"}


def neighbour_map(registry):
    """For each station, its upstream and downstream station IDs, one and two hops out.

    Returns four dicts (up1, up2, down1, down2). A missing neighbour is None.
    """
    up1, up2, down1, down2 = {}, {}, {}, {}
    for _, group in registry.sort_values("postmile").groupby(["freeway", "direction"]):
        ids = list(group["station_id"])
        lower_is_upstream = group["direction"].iloc[0] in UPSTREAM_IF_LOWER_POSTMILE
        for i, sid in enumerate(ids):
            # step -1 is the station with the next lower postmile, +1 the next higher.
            before = {1: _at(ids, i - 1), 2: _at(ids, i - 2)}
            after = {1: _at(ids, i + 1), 2: _at(ids, i + 2)}
            upstream, downstream = (before, after) if lower_is_upstream else (after, before)
            up1[sid], up2[sid] = upstream[1], upstream[2]
            down1[sid], down2[sid] = downstream[1], downstream[2]
    return up1, up2, down1, down2


def _at(ids, i):
    """The ID at position i, or None when i is off either end of the list."""
    return ids[i] if 0 <= i < len(ids) else None


def add_neighbour_features(df, registry):
    """Speed and ratio of the nearest neighbours at the same timestamp, and one step of lag."""
    up1, up2, down1, down2 = neighbour_map(registry)
    speed_wide = df.pivot(index="timestamp", columns="station_id", values="speed")
    ratio_wide = df.pivot(index="timestamp", columns="station_id", values="speed_ratio")

    def series_for(wide, sid):
        if sid is None or sid not in wide.columns:
            return np.full(len(wide), np.nan)
        return wide[sid].to_numpy()

    pieces = []
    for sid, group in df.groupby("station_id", observed=True, sort=False):
        idx = pd.DatetimeIndex(group["timestamp"])
        frame = pd.DataFrame(index=group.index)
        for name, mapping in [("up1", up1), ("up2", up2), ("down1", down1), ("down2", down2)]:
            neighbour = mapping.get(sid)
            frame[f"{name}_speed"] = pd.Series(series_for(speed_wide, neighbour), index=speed_wide.index).reindex(idx).to_numpy()
        up_speed = pd.Series(series_for(speed_wide, up1.get(sid)), index=speed_wide.index)
        down_speed = pd.Series(series_for(speed_wide, down1.get(sid)), index=speed_wide.index)
        frame["up1_speed_lag_3"] = up_speed.shift(3).reindex(idx).to_numpy()
        frame["down1_speed_lag_3"] = down_speed.shift(3).reindex(idx).to_numpy()
        frame["up1_ratio"] = pd.Series(series_for(ratio_wide, up1.get(sid)), index=ratio_wide.index).reindex(idx).to_numpy()
        pieces.append(frame)
    return pd.concat(pieces).reindex(df.index)


def add_weather(df, weather):
    """Join hourly precipitation (UTC timestamp, precip_mm) onto each row's hour."""
    hourly = weather.assign(hour=pd.to_datetime(weather["timestamp"], utc=True)).set_index("hour")["precip_mm"]
    hourly = hourly.sort_index()
    hour = df["timestamp"].dt.floor("h")
    df["precip_mm_1h"] = hourly.reindex(hour).to_numpy().astype("float32")
    # Three-hour total from the hours before and including this one. Assumes the
    # weather file has one row per hour with no gaps.
    df["precip_mm_3h"] = hourly.rolling(3, min_periods=1).sum().reindex(hour).to_numpy().astype("float32")
    df["is_raining"] = (df["precip_mm_1h"] > 0).astype("int8")
    return df


def build_features(df, registry, incidents, weather=None):
    """Return the feature table and its feature column names.

    df must be the labelled grid, sorted by station_id and timestamp, with a fresh
    0..n-1 index. It needs columns station_id, timestamp, speed, occupancy, flow,
    speed_ratio, profile_speed, label, dow and is_holiday. incidents is the output
    of incidents.join_incidents on that same sorted grid, so rows line up by position.
    """
    df = pd.concat([df.reset_index(drop=True), incidents.reset_index(drop=True)], axis=1)

    by_station = df.groupby("station_id", observed=True, sort=False)
    df["speed_lag_0"] = df["speed"]
    for k in SPEED_LAGS:
        df[f"speed_lag_{k}"] = by_station["speed"].shift(k)
    df["occ_lag_0"] = df["occupancy"]
    df["occ_lag_3"] = by_station["occupancy"].shift(3)
    df["flow_lag_0"] = df["flow"]
    df["flow_lag_3"] = by_station["flow"].shift(3)
    df["speed_roll_mean_30"] = by_station["speed"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    df["speed_roll_mean_60"] = by_station["speed"].transform(lambda s: s.rolling(12, min_periods=1).mean())
    df["speed_roll_std_30"] = by_station["speed"].transform(lambda s: s.rolling(6, min_periods=2).std())
    df["speed_slope_15"] = df["speed_lag_0"] - df["speed_lag_3"]

    neighbours = add_neighbour_features(df, registry)
    df = pd.concat([df, neighbours], axis=1)

    minute = df["timestamp"].dt.tz_convert("America/Los_Angeles")
    minute_of_day = (minute.dt.hour * 60 + minute.dt.minute).to_numpy()
    df["minute_sin"] = np.sin(2 * np.pi * minute_of_day / 1440).astype("float32")
    df["minute_cos"] = np.cos(2 * np.pi * minute_of_day / 1440).astype("float32")
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7).astype("float32")
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7).astype("float32")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")

    if weather is not None:
        df = add_weather(df, weather)

    for name, steps in HORIZONS.items():
        df[name] = by_station["label"].shift(-steps)

    feature_columns = [
        "speed_lag_0", *[f"speed_lag_{k}" for k in SPEED_LAGS],
        "occ_lag_0", "occ_lag_3", "flow_lag_0", "flow_lag_3",
        "speed_roll_mean_30", "speed_roll_mean_60", "speed_roll_std_30", "speed_slope_15",
        "profile_speed", "speed_ratio",
        "up1_speed", "up2_speed", "down1_speed", "down2_speed",
        "up1_speed_lag_3", "down1_speed_lag_3", "up1_ratio",
        "incident_active", "incident_severity", "minutes_since_incident_start", "nearest_incident_miles",
        "minute_sin", "minute_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday",
    ]
    if weather is not None:
        feature_columns += ["precip_mm_1h", "precip_mm_3h", "is_raining"]

    df = df.dropna(subset=TARGET_COLUMNS)
    return df, feature_columns
