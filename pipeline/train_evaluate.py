"""Stages 10 and 11: LightGBM per horizon, calibration, and test scoring.

Train: fit one LightGBM model per horizon on the training months only. The
number of trees is fixed (not early-stopped), so the calibration and test
months are never used to pick the model.

Calibrate: fit an isotonic correction on the calibration months, so that a
score of 0.3 means about 30%.

Test: score the calibrated models on the test months, against two baselines:
  - "traffic stays the same": predict the current 3-step label
  - the ratio rule: predict slow when speed is below 70% of the profile
"""
import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PARAMS = {
    "objective": "binary",
    "is_unbalance": True,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "verbose": -1,
}
NUM_TREES = 300
SUCCESS_AP_MULTIPLE = 2.0  # plan: 30-minute AP at least twice persistence


def train_model(train_df, features, target, num_trees=NUM_TREES):
    """Fit one LightGBM binary classifier for one horizon."""
    data = lgb.Dataset(train_df[features], label=train_df[target].astype(int))
    return lgb.train(PARAMS, data, num_boost_round=num_trees)


def fit_calibrator(raw_scores, labels):
    """Isotonic map from raw model scores to calibrated probabilities."""
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_scores, labels)
    return calibrator


def score_metrics(labels, scores):
    """AP, ROC-AUC and Brier score. ROC-AUC is None if one class is absent."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    has_both = labels.min() != labels.max()
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)) if has_both else None,
        "brier": float(brier_score_loss(labels, np.clip(scores, 0, 1))),
        "positive_rate": float(labels.mean()),
    }


def evaluate_horizon(model, calibrator, test_df, features, target):
    """Compare the calibrated model with the baselines on the test months."""
    labels = test_df[target].astype(int).to_numpy()
    model_scores = calibrator.predict(model.predict(test_df[features]))

    # Baseline 1: traffic stays the same. The current label is the prediction.
    persistence = test_df["label"].to_numpy()
    # Baseline 2: the ratio rule, slow when speed is below 70% of profile.
    ratio_rule = (test_df["speed_ratio"] < 0.7).to_numpy().astype(float)

    result = {
        "model": score_metrics(labels, model_scores),
        "persistence": score_metrics(labels, persistence),
        "ratio_rule": score_metrics(labels, ratio_rule),
    }
    return result


def check_success(report):
    """Return the plan's first success check, for the 30-minute horizon."""
    model_ap = report["y_30"]["model"]["average_precision"]
    persistence_ap = report["y_30"]["persistence"]["average_precision"]
    passed = model_ap >= SUCCESS_AP_MULTIPLE * persistence_ap
    return {
        "rule": "30-minute AP at least 2x persistence",
        "model_ap": model_ap,
        "persistence_ap": persistence_ap,
        "passed": bool(passed),
    }
