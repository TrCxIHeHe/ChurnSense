import json

import pytest

from src.experiment_tracker import experiments_dataframe, load_experiments
from src.prediction import ModelNotFoundError, load_metadata, load_model


def test_model_artifacts_saved(trained):
    models = trained["models_dir"]
    assert (models / "current.json").exists()
    assert list(models.glob("churnsense_v*.joblib"))
    assert list(models.glob("churnsense_v*_metadata.json"))


def test_model_loads_and_has_predict_proba(trained):
    model = load_model(trained["models_dir"])
    assert hasattr(model, "predict_proba")


def test_missing_model_raises_clean_error(tmp_path):
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path)


def test_metadata_contents(trained):
    meta = load_metadata(trained["models_dir"])
    for key in ("model_version", "training_date", "selected_algorithm", "dataset", "test_metrics"):
        assert key in meta
    assert meta["selected_algorithm"] in {"Logistic Regression", "Random Forest", "Gradient Boosting"}
    assert 0.5 < meta["test_metrics"]["roc_auc"] <= 1.0


def test_selection_uses_cv_not_test(trained):
    metrics = json.loads((trained["artifacts_dir"] / "metrics.json").read_text())
    best_cv = max(metrics["models"], key=lambda n: metrics["models"][n]["cv_roc_auc"])
    assert metrics["selected_model"] == best_cv


def test_all_three_models_trained_and_tracked(trained):
    exps = load_experiments(trained["artifacts_dir"])
    assert {e["model_name"] for e in exps} == {"Logistic Regression", "Random Forest", "Gradient Boosting"}
    assert sum(e["selected"] for e in exps) == 1
    df = experiments_dataframe(trained["artifacts_dir"])
    assert len(df) == 3 and df["experiment_id"].is_unique


def test_reference_stats_created(trained):
    assert (trained["artifacts_dir"] / "reference_stats.json").exists()
