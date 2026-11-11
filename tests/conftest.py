import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data_loader import load_raw_data  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    return load_raw_data(config.DATA_FILE)


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train a fast model into temporary dirs (does not touch project models/)."""
    from src.training import run_training

    base = tmp_path_factory.mktemp("run")
    models, artifacts = base / "models", base / "artifacts"
    out = run_training(models_dir=models, artifacts_dir=artifacts, n_iter=1, cv_folds=2,
                       save_plots=False)
    return {"models_dir": models, "artifacts_dir": artifacts, "out": out}


@pytest.fixture()
def sample_customer():
    return {
        "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No",
        "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5, "TotalCharges": 447.5,
    }
