"""Clean PeMS station_5min files into one Parquet table per month.

Reads the district files in data/pems_raw/, keeps only the Fremont stations,
converts local Pacific time to UTC, nulls impossible or imputed readings,
and puts every station on a full 5-minute grid. Writes
data/pems_processed/YYYY-MM.parquet, one row per station per 5 minutes.

ASSUMPTIONS TO CHECK ON THE FIRST REAL FILE (plan, stage 4, step 2):
  - the column order below (no header row, first 12 columns used)
  - the timestamp format, MM/DD/YYYY HH:MM:SS
  - occupancy is a 0 to 1 fraction, as PeMS documents it

Usage:
    python pipeline/clean_pems.py --stations data/pems_meta/fremont_stations.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA, MIN_PCT_OBSERVED, OCC_MAX, SPEED_MAX_MPH

RAW_DIR = DATA / "pems_raw"
OUT_DIR = DATA / "pems_processed"

# First 12 columns of station_5min, in order. Later columns are per-lane and unused.
COLUMNS = [
    "timestamp", "station_id", "district", "freeway", "direction", "lane_type",
    "length", "samples", "pct_observed", "flow", "occupancy", "speed",
]
TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S"
LOCAL_TZ = "America/Los_Angeles"
MAX_FILL_STEPS = 3  # gaps of up to 3 five-minute steps are forward-filled

FILE_RE = re.compile(r"d04_text_station_5min_(\d{4})_(\d{2})_\d{2}\.txt\.gz$")


def load_station_ids(path):
    """Read the Fremont station IDs, one column named station_id."""
    return set(pd.read_csv(path)["station_id"].astype(int))


def read_station_file(path, station_ids, chunksize=200_000):
    """Stream one gzip file in chunks, keeping only the wanted stations."""
    kept = []
    reader = pd.read_csv(path, header=None, names=COLUMNS, usecols=range(12), chunksize=chunksize)
    for chunk in reader:
        kept.append(chunk[chunk["station_id"].isin(station_ids)])
    if not kept:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(kept, ignore_index=True)


def clean(raw):
    """Apply timezone, imputation and bounds rules. Returns one row per reading."""
    df = raw.copy()

    local = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FORMAT)
    # Clocks go back once a year, so 1:00 to 2:00 am happens twice. Drop those rows
    # rather than guess which copy is right. Spring-forward gaps shift forward.
    df["timestamp"] = (
        local.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    df = df.dropna(subset=["timestamp"])

    # PeMS fills gaps in detector data with estimates. Drop those rows.
    imputed = df["pct_observed"] < MIN_PCT_OBSERVED
    df.loc[imputed, ["speed", "occupancy", "flow"]] = np.nan
    df["observed_frac"] = df["pct_observed"] / 100.0

    df.loc[(df["speed"] < 0) | (df["speed"] > SPEED_MAX_MPH), "speed"] = np.nan
    df.loc[(df["occupancy"] < 0) | (df["occupancy"] > OCC_MAX), "occupancy"] = np.nan
    df.loc[df["flow"] < 0, "flow"] = np.nan

    df = df.drop_duplicates(subset=["station_id", "timestamp"], keep="last")
    return df[["station_id", "timestamp", "speed", "occupancy", "flow", "observed_frac", "samples"]]


def to_grid(df):
    """Put every station on one full 5-minute UTC grid, with short gaps filled."""
    grid = pd.date_range(df["timestamp"].min().floor("5min"), df["timestamp"].max(), freq="5min", tz="UTC")
    pieces = []
    for station_id, group in df.groupby("station_id", observed=True):
        frame = group.set_index("timestamp").reindex(grid)
        frame["station_id"] = station_id
        frame["observed_frac"] = frame["observed_frac"].fillna(0.0)
        frame["samples"] = frame["samples"].fillna(0)
        for col in ["speed", "occupancy", "flow"]:
            frame[col] = frame[col].ffill(limit=MAX_FILL_STEPS)
        frame.index.name = "timestamp"
        pieces.append(frame.reset_index())
    out = pd.concat(pieces, ignore_index=True)
    out["station_id"] = out["station_id"].astype("category")
    for col in ["speed", "occupancy", "flow", "observed_frac"]:
        out[col] = out[col].astype("float32")
    out["samples"] = out["samples"].astype("int16")
    return out[["station_id", "timestamp", "speed", "occupancy", "flow", "observed_frac", "samples"]]


def clean_month(files, station_ids):
    """Read the given day files for one month and return the gridded table."""
    raw = pd.concat([read_station_file(f, station_ids) for f in files], ignore_index=True)
    if raw.empty:
        raise ValueError("no rows for the chosen stations in these files")
    return to_grid(clean(raw))


def group_by_month(raw_dir):
    """Map 'YYYY-MM' to the list of day files in that month."""
    months = {}
    for path in sorted(raw_dir.glob("*.txt.gz")):
        match = FILE_RE.search(path.name)
        if match:
            key = f"{match.group(1)}-{match.group(2)}"
            months.setdefault(key, []).append(path)
    return months


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stations", type=Path, required=True, help="CSV with a station_id column")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    station_ids = load_station_ids(args.stations)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for month, files in group_by_month(args.raw_dir).items():
        table = clean_month(files, station_ids)
        out = args.out_dir / f"{month}.parquet"
        table.to_parquet(out, index=False)
        print(f"{month}: {len(files)} day files, {len(table):,} rows -> {out}")


if __name__ == "__main__":
    main()
