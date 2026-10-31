"""Stage 6: typical speed profile and the slow-episode label.

The profile is the typical speed for each station, 5-minute slot of the local
day, and day type. It is built from training rows only, so the test months
cannot leak into the labels. A station is "slow" at a step when its speed is
below 60% of that profile. The label is 1 when the station is slow for three
steps in a row, which is 15 minutes, ending at this step.
"""
import holidays
import numpy as np
import pandas as pd

LOCAL_TZ = "America/Los_Angeles"
SLOW_RATIO = 0.6
SLOW_STEPS = 3          # slow for this many steps in a row
MIN_CELL_COUNT = 20     # cells with fewer training samples fall back to the hourly median
SMOOTH_SLOTS = 7        # centered rolling median over neighbouring 5-minute slots
SLOW_RATIO_RULE = 0.7   # the simple "ratio rule" baseline


def add_local_time(df, state="CA"):
    """Add the local 5-minute slot of day and the day type (weekday, saturday, sunday_holiday)."""
    local = df["timestamp"].dt.tz_convert(LOCAL_TZ)
    years = sorted(set(local.dt.year))
    holiday_dates = holidays.US(state=state, years=years)

    df["slot"] = (local.dt.hour * 12 + local.dt.minute // 5).astype("int16")
    # Check each calendar date once, not each of the millions of rows.
    dates = local.dt.date
    is_holiday = dates.isin(list(holiday_dates.keys())).to_numpy()
    dow = local.dt.dayofweek.to_numpy()
    day_type = np.where(is_holiday | (dow == 6), "sunday_holiday", np.where(dow == 5, "saturday", "weekday"))
    df["day_type"] = day_type
    df["is_holiday"] = is_holiday.astype("int8")
    df["dow"] = dow.astype("int8")
    return df


def build_profile(df, train_mask):
    """Typical speed per (station, slot, day type), from training rows only."""
    train = df.loc[train_mask, ["station_id", "slot", "day_type", "speed"]].dropna(subset=["speed"])
    keys = ["station_id", "slot", "day_type"]
    cells = train.groupby(keys, observed=True)["speed"].agg(["median", "size"]).reset_index()

    hourly = train.assign(hour=train["slot"] // 12).groupby(["station_id", "hour", "day_type"], observed=True)
    hourly_median = hourly["speed"].median().rename("hour_median").reset_index()

    # Smooth each station's slot profile with a centered rolling median across slots.
    cells = cells.sort_values(["station_id", "day_type", "slot"])
    cells["smoothed"] = (
        cells.groupby(["station_id", "day_type"], observed=True)["median"]
        .transform(lambda s: s.rolling(SMOOTH_SLOTS, center=True, min_periods=1).median())
    )
    cells["hour"] = cells["slot"] // 12
    cells = cells.merge(hourly_median, on=["station_id", "hour", "day_type"], how="left")
    cells["profile_speed"] = np.where(cells["size"] >= MIN_CELL_COUNT, cells["smoothed"], cells["hour_median"])
    return cells[keys + ["profile_speed"]]


def add_labels(df, profile):
    """Attach the profile, speed ratio, slow flag and 3-step label. df must be sorted by station and time."""
    df = df.merge(profile, on=["station_id", "slot", "day_type"], how="left")
    df["speed_ratio"] = (df["speed"] / df["profile_speed"]).astype("float32")
    # A missing speed is unknown, not "not slow", so it stays NaN and breaks the run of 3.
    df["is_slow"] = (df["speed_ratio"] < SLOW_RATIO).astype("float32").where(df["speed_ratio"].notna())
    df["label"] = (
        df.groupby("station_id", observed=True)["is_slow"]
        .transform(lambda s: s.rolling(SLOW_STEPS).sum())
        .eq(SLOW_STEPS)
        .astype("int8")
    )
    return df
