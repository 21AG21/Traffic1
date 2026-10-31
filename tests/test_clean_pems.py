"""Tests for clean_pems.py, using a few hand-written rows (no random data)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from clean_pems import clean, read_station_file, to_grid  # noqa: E402


def raw_rows(rows):
    """Build a frame shaped like read_station_file's output."""
    cols = ["timestamp", "station_id", "pct_observed", "flow", "occupancy", "speed", "samples"]
    frame = pd.DataFrame(rows, columns=cols)
    frame["district"] = 4
    frame["freeway"] = 880
    frame["direction"] = "N"
    frame["lane_type"] = "ML"
    frame["length"] = 0.5
    return frame


def test_local_time_converts_to_utc():
    df = clean(raw_rows([("01/15/2026 08:00:00", 101, 100, 1000, 0.1, 65, 20)]))
    assert df["timestamp"].iloc[0] == pd.Timestamp("2026-01-15 16:00", tz="UTC")


def test_ambiguous_fall_back_hour_is_dropped():
    # Nov 1 2026 1:30 am happens twice in Los Angeles, so the row is dropped.
    df = clean(raw_rows([
        ("11/01/2026 01:30:00", 101, 100, 1000, 0.1, 65, 20),
        ("11/01/2026 03:00:00", 101, 100, 1000, 0.1, 65, 20),
    ]))
    assert len(df) == 1
    assert df["timestamp"].iloc[0] == pd.Timestamp("2026-11-01 11:00", tz="UTC")


def test_imputed_rows_are_nulled_and_kept_as_zero_observed():
    df = clean(raw_rows([("01/15/2026 08:00:00", 101, 30, 1000, 0.1, 65, 20)]))
    assert np.isnan(df["speed"].iloc[0])
    assert np.isnan(df["flow"].iloc[0])
    assert df["observed_frac"].iloc[0] == pytest.approx(0.3)


def test_impossible_values_are_nulled():
    df = clean(raw_rows([
        ("01/15/2026 08:00:00", 101, 100, 1000, 0.1, 150, 20),   # speed over 100 mph
        ("01/15/2026 08:05:00", 101, 100, 1000, 1.7, 60, 20),    # occupancy over 1
        ("01/15/2026 08:10:00", 101, 100, -5, 0.1, 60, 20),      # negative flow
    ]))
    assert np.isnan(df["speed"].iloc[0])
    assert np.isnan(df["occupancy"].iloc[1])
    assert np.isnan(df["flow"].iloc[2])


def test_grid_fills_short_gaps_and_marks_them_unobserved():
    rows = [
        ("01/15/2026 08:00:00", 101, 100, 1000, 0.1, 60, 20),
        ("01/15/2026 08:05:00", 101, 100, 1000, 0.1, 62, 20),
        # 08:10 and 08:15 are missing; the grid should show them.
        ("01/15/2026 08:20:00", 101, 100, 1000, 0.1, 64, 20),
    ]
    out = to_grid(clean(raw_rows(rows)))
    assert len(out) == 5
    gap = out.iloc[2]  # 08:10 local, 16:10 UTC
    assert gap["speed"] == np.float32(62)  # forward-filled from 08:05
    assert gap["observed_frac"] == 0.0
    assert gap["samples"] == 0


def test_long_gap_is_not_filled():
    rows = [
        ("01/15/2026 08:00:00", 101, 100, 1000, 0.1, 60, 20),
        ("01/15/2026 09:00:00", 101, 100, 1000, 0.1, 60, 20),
    ]
    out = to_grid(clean(raw_rows(rows)))
    # An hour is 11 missing steps, more than the 3-step limit, so the middle stays empty.
    middle = out.iloc[6]
    assert np.isnan(middle["speed"])


def test_read_station_file_keeps_only_wanted_stations(tmp_path):
    path = tmp_path / "d04_text_station_5min_2026_01_15.txt.gz"
    lines = [
        "01/15/2026 08:00:00,101,4,880,N,ML,0.5,20,100,1000,0.1,65,1,2,3,4,5",
        "01/15/2026 08:00:00,999,4,880,N,ML,0.5,20,100,1000,0.1,65,1,2,3,4,5",
    ]
    import gzip
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")
    out = read_station_file(path, {101})
    assert list(out["station_id"]) == [101]
