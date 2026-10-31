"""Run stages 5 to 11 end to end: labels, incidents, features, split, train, calibrate, evaluate.

Inputs (all produced by earlier steps or by you):
  data/pems_processed/YYYY-MM.parquet     from clean_pems.py
  data/pems_meta/fremont_stations.csv     station registry: station_id, freeway, direction, postmile, lat, lon
  data/fremont_events.csv                 from collect_511.py
  data/weather_hourly.csv                 optional: timestamp (UTC), precip_mm

Outputs:
  models/gbt_y_15.txt, gbt_y_30.txt, gbt_y_60.txt
  models/calibration.json                 isotonic knots per horizon
  output/evaluation_report.json           test-month scores against the baselines

Usage:
    python pipeline/run_pipeline.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA, ROOT
from features import TARGET_COLUMNS, build_features
from incidents import join_incidents
from labels import add_labels, add_local_time, build_profile
from split import split_boundaries, split_masks
from train_evaluate import check_success, evaluate_horizon, fit_calibrator, train_model

MODELS = ROOT / "models"
OUTPUT = ROOT / "output"
CATEGORICAL = ["station_id", "freeway", "direction"]


def load_pems():
    files = sorted((DATA / "pems_processed").glob("*.parquet"))
    if not files:
        raise SystemExit("no monthly files in data/pems_processed/. Run clean_pems.py first.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)


def load_registry():
    registry = pd.read_csv(DATA / "pems_meta" / "fremont_stations.csv")
    registry["station_id"] = registry["station_id"].astype(int)
    registry["direction"] = registry["direction"].astype(str).str.upper().str[:1]
    return registry


def load_optional_weather():
    path = DATA / "weather_hourly.csv"
    return pd.read_csv(path) if path.exists() else None


def main():
    registry = load_registry()
    grid = load_pems()
    grid = add_local_time(grid)
    print(f"{len(grid):,} rows, {grid['station_id'].nunique()} stations")

    cal_start, test_start = split_boundaries(grid["timestamp"])
    train_mask, _, _ = split_masks(grid["timestamp"], cal_start, test_start)

    profile = build_profile(grid, train_mask)
    grid = add_labels(grid, profile)

    events = pd.read_csv(DATA / "fremont_events.csv")
    incidents = join_incidents(grid, registry, events)

    weather = load_optional_weather()
    feats, features = build_features(grid, registry, incidents, weather)
    feats = feats.merge(registry[["station_id", "freeway", "direction"]], on="station_id", how="left")
    for col in ["freeway", "direction"]:
        feats[col] = feats[col].astype("category")
    features = features + CATEGORICAL

    train_df, cal_df, test_df = _split_feats(feats, cal_start, test_start)
    print(f"train {len(train_df):,}  calibrate {len(cal_df):,}  test {len(test_df):,}")

    MODELS.mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)
    report, calibration = {}, {}
    for target in TARGET_COLUMNS:
        model = train_model(train_df, features, target)
        model.save_model(str(MODELS / f"gbt_{target}.txt"))

        cal_raw = model.predict(cal_df[features])
        calibrator = fit_calibrator(cal_raw, cal_df[target].astype(int))
        calibration[target] = {
            "x": calibrator.X_thresholds_.tolist(),
            "y": calibrator.y_thresholds_.tolist(),
        }
        report[target] = evaluate_horizon(model, calibrator, test_df, features, target)
        print(target, "model AP", round(report[target]["model"]["average_precision"], 4))

    report["success_check"] = check_success(report)
    report["data"] = {
        "train_rows": int(len(train_df)),
        "calibrate_rows": int(len(cal_df)),
        "test_rows": int(len(test_df)),
        "calibration_starts": str(cal_start),
        "test_starts": str(test_start),
        "weather_used": weather is not None,
        "labelled_as": "SYNTHETIC" if (DATA / "SYNTHETIC").exists() else "REAL",
    }
    (MODELS / "calibration.json").write_text(json.dumps(calibration, indent=2))
    (OUTPUT / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    print("success check:", report["success_check"])


def _split_feats(feats, cal_start, test_start):
    train, cal, test = split_masks(feats["timestamp"], cal_start, test_start)
    return feats[train], feats[cal], feats[test]


if __name__ == "__main__":
    main()
