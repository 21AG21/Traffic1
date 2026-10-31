"""Stage 9: chronological split by month, with a one-day gap between blocks.

With 12 months of data: the first 8 train, the next 2 calibrate, the last 2 test.
The same 8:2:2 ratio is used for any number of months, with at least one month
in each block. A one-day gap (288 steps) is dropped before each new block, so
lags and 60-minute targets never straddle a boundary.
"""
import numpy as np
import pandas as pd

GAP = pd.Timedelta("1D")
TRAIN_SHARE, CAL_SHARE = 8 / 12, 2 / 12


def split_boundaries(timestamps):
    """Return (cal_start, test_start) as UTC timestamps, chosen by calendar month."""
    months = sorted(pd.DatetimeIndex(timestamps).tz_convert("UTC").to_period("M").unique())
    n = len(months)
    n_train = max(1, round(n * TRAIN_SHARE))
    n_cal = max(1, round(n * CAL_SHARE))
    if n - n_train - n_cal < 1:
        raise ValueError(f"need more months: found {n}, need at least 5 for an 8/2/2 split")
    cal_start = months[n_train].start_time.tz_localize("UTC")
    test_start = months[n_train + n_cal].start_time.tz_localize("UTC")
    return cal_start, test_start


def split_masks(timestamps, cal_start, test_start):
    """Boolean arrays for train, calibration and test rows, with the gaps removed."""
    ts = pd.DatetimeIndex(timestamps)
    train = ts < (cal_start - GAP)
    cal = (ts >= cal_start) & (ts < (test_start - GAP))
    test = ts >= test_start
    return np.asarray(train), np.asarray(cal), np.asarray(test)
