# Fremont bottleneck pipeline

Stages 1 to 4 of `21_PLAN_M1_TRAFFIC_MODEL.md`. Run everything from the repo root.

## Setup (Mac)

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your logins in `.env` (see the plan). It is ignored by git.

## Stage 1 and 2: 511 events

One poll:

```
python pipeline/collect_511.py
```

Schedule it every 15 minutes with launchd, not cron. Output:
- `data/events_raw/YYYY-MM-DD.jsonl`: every raw poll, so filters can change later.
- `data/fremont_events.csv`: one row per event, deduplicated by `event_id`.

## Stage 4: PeMS cleaning

Put the downloaded `.txt.gz` day files in `data/pems_raw/`, and a CSV with a `station_id` column in `data/pems_meta/fremont_stations.csv`. Then:

```
python pipeline/clean_pems.py --stations data/pems_meta/fremont_stations.csv
```

Output: `data/pems_processed/YYYY-MM.parquet`, one row per station per 5 minutes.

## Tests

```
python -m pytest tests/
```

## Not verified yet

These assumptions need checking on the first real PeMS file and the first real 511 response:
- the PeMS column order and the timestamp format (`clean_pems.py`)
- the 511 JSON shape: that the events list is under `events` and roads have a `direction` field (`collect_511.py`)
- the station registry: `fremont_stations.csv` is not built yet. The plan builds it from the PeMS meta file.
