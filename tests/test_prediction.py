import pandas as pd
import pytest

from src import config
from src.data_loader import load_raw_data
from src.insights import generate_insights
from src.prediction import load_model, predict_batch, predict_single, risk_category


@pytest.fixture(scope="module")
def model(trained):
    return load_model(trained["models_dir"])


def test_single_prediction_output(model, sample_customer, trained):
    res = predict_single(model, sample_customer, artifacts_dir=trained["artifacts_dir"])
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["churn_probability"] + res["retention_probability"] == pytest.approx(1.0)
    assert res["risk_category"] in {"Low Risk", "Medium Risk", "High Risk"}
    assert res["predicted_churn"] == (res["churn_probability"] >= 0.5)


def test_single_prediction_invalid_input_rejected(model, sample_customer, trained):
    bad = dict(sample_customer)
    bad["tenure"] = -5
    with pytest.raises(ValueError):
        predict_single(model, bad, artifacts_dir=trained["artifacts_dir"])


def test_high_risk_profile_scores_higher_than_loyal_profile(model, sample_customer, trained):
    loyal = dict(sample_customer, tenure=60, Contract="Two year", InternetService="DSL",
                 TechSupport="Yes", OnlineSecurity="Yes", PaymentMethod="Credit card (automatic)",
                 MonthlyCharges=55.0, TotalCharges=3300.0)
    a = predict_single(model, sample_customer, artifacts_dir=trained["artifacts_dir"])
    b = predict_single(model, loyal, artifacts_dir=trained["artifacts_dir"])
    assert a["churn_probability"] > b["churn_probability"]


def test_batch_prediction(model, trained):
    df = load_raw_data(config.DATA_FILE).drop(columns=["Churn"]).head(50)
    out, report = predict_batch(model, df, artifacts_dir=trained["artifacts_dir"])
    assert len(out) == 50 and report["invalid_rows"] == 0
    assert out["churn_probability"].between(0, 1).all()
    assert set(out["risk_category"]) <= {"Low Risk", "Medium Risk", "High Risk"}
    assert set(out["predicted_churn"]) <= {"Yes", "No"}


def test_batch_missing_columns_raises(model, trained):
    with pytest.raises(ValueError):
        predict_batch(model, pd.DataFrame({"tenure": [1]}), artifacts_dir=trained["artifacts_dir"])


def test_batch_skips_invalid_rows(model, trained):
    df = load_raw_data(config.DATA_FILE).drop(columns=["Churn"]).head(5).copy()
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(object)
    df.loc[0, "MonthlyCharges"] = "not-a-number"
    out, report = predict_batch(model, df, artifacts_dir=trained["artifacts_dir"])
    assert len(out) == 4 and report["invalid_rows"] == 1


def test_batch_handles_blank_total_charges(model, trained):
    df = load_raw_data(config.DATA_FILE).drop(columns=["Churn"]).head(3).copy()
    df["TotalCharges"] = df["TotalCharges"].astype(object)
    df.loc[0, "TotalCharges"] = " "
    out, _ = predict_batch(model, df, artifacts_dir=trained["artifacts_dir"])
    assert len(out) == 3


def test_risk_category_thresholds():
    assert risk_category(0.05) == "Low Risk"
    assert risk_category(0.45) == "Medium Risk"
    assert risk_category(0.90) == "High Risk"


def test_insights_rules(sample_customer):
    factors = {i["factor"] for i in generate_insights(sample_customer, "High Risk")}
    assert {"Month-to-month contract", "Short tenure", "Electronic check payment"} <= factors
