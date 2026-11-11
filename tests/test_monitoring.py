import numpy as np
import pandas as pd

from src import config
from src.data_loader import clean_data, load_raw_data
from src.monitoring import (build_reference_stats, compute_drift, drift_summary, log_predictions,
                            prediction_stats, psi_from_proportions, validate_input)


def _train_X():
    X, _, _ = clean_data(load_raw_data(config.DATA_FILE))
    return X


def test_psi_identical_is_zero_and_shift_is_large():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert psi_from_proportions(p, p) == 0.0
    assert psi_from_proportions(p, np.array([0.7, 0.1, 0.1, 0.1])) > 0.25


def test_no_drift_on_same_distribution():
    X = _train_X()
    ref = build_reference_stats(X)
    drift = compute_drift(X, ref)
    assert drift["psi"].max() < 0.01
    assert drift_summary(drift)["overall"] == "Stable"


def test_drift_detected_when_data_shifted():
    X = _train_X()
    ref = build_reference_stats(X)
    shifted = X.copy()
    shifted["MonthlyCharges"] = shifted["MonthlyCharges"] * 2 + 50
    shifted["Contract"] = "Month-to-month"
    drift = compute_drift(shifted, ref)
    summary = drift_summary(drift)
    assert summary["overall"] == "Significant drift"
    flagged = drift[drift["psi"] >= config.PSI_SIGNIFICANT]["feature"].tolist()
    assert "MonthlyCharges" in flagged and "Contract" in flagged
    assert summary["warnings"]


def test_validate_input_reports_missing_columns():
    rep = validate_input(pd.DataFrame({"tenure": [1]}))
    assert rep["errors"] and "Contract" in rep["missing_columns"]


def test_validate_input_flags_unexpected_values():
    df = load_raw_data(config.DATA_FILE).drop(columns=["Churn"]).head(5).copy()
    df.loc[0, "Contract"] = "Five year"
    rep = validate_input(df)
    assert not rep["errors"]
    assert any("Contract" in w for w in rep["warnings"])


def test_prediction_stats_from_log(tmp_path):
    assert prediction_stats(tmp_path)["count"] == 0
    log_predictions([0.1, 0.5, 0.9, 0.95], "batch", tmp_path)
    stats = prediction_stats(tmp_path)
    assert stats["count"] == 4 and stats["high_risk"] == 2
    assert sum(stats["histogram"]["counts"]) == 4
