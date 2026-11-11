import pandas as pd
import pytest

from src import config
from src.data_loader import clean_data, load_raw_data, normalize_features


def test_raw_data_loads_with_expected_columns(raw_df):
    assert len(raw_df) > 7000
    for col in config.FEATURE_COLUMNS + [config.TARGET, config.ID_COLUMN]:
        assert col in raw_df.columns


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "nope.csv", auto_download=False)


def test_missing_required_columns_raises(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_raw_data(p, auto_download=False)


def test_clean_data_outputs(raw_df):
    X, y, report = clean_data(raw_df)
    assert config.ID_COLUMN not in X.columns
    assert list(X.columns) == config.FEATURE_COLUMNS
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y) == report["rows_clean"]
    assert report["rows_clean"] <= report["rows_raw"]
    assert X["TotalCharges"].dtype.kind == "f"
    assert 0.1 < y.mean() < 0.5


def test_total_charges_blank_handled(raw_df):
    # The real IBM data has blank TotalCharges for tenure == 0 customers
    X, _, report = clean_data(raw_df)
    assert report["total_charges_missing"] >= 1
    assert X.loc[X["tenure"] == 0, "TotalCharges"].fillna(0).eq(0).all()


def test_normalize_senior_citizen_numeric_to_yes_no():
    df = pd.DataFrame({"SeniorCitizen": [0, 1]})
    assert normalize_features(df)["SeniorCitizen"].tolist() == ["No", "Yes"]
