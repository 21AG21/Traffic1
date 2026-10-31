"""Shared settings for the Fremont bottleneck pipeline."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ENV_FILE = ROOT / ".env"

# Bounding box for Fremont, from the plan (stage 2).
LAT_MIN, LAT_MAX = 37.447, 37.612
LON_MIN, LON_MAX = -122.075, -121.888

# An event counts as Fremont if a road name matches, or its point is in the box.
FREMONT_ROAD_RE = re.compile(r"\b(?:I-?880|I-?680|SR-?84)\b", re.IGNORECASE)

# Speed and occupancy bounds used to null impossible PeMS readings (stage 4).
SPEED_MAX_MPH = 100.0
OCC_MAX = 1.0

# A PeMS row with % Observed below this came from the imputer, so it is dropped.
MIN_PCT_OBSERVED = 50.0
