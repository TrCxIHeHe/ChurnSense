"""Central configuration for ChurnSense."""
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

DATA_FILE = DATA_DIR / "Telco-Customer-Churn.csv"
DATA_SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATA_SOURCE_PAGE = "https://github.com/IBM/telco-customer-churn-on-icp4d"
DATA_SOURCE_NAME = "IBM Telco Customer Churn (IBM Cloud Pak for Data sample dataset)"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER_SEARCH = 12

TARGET = "Churn"
ID_COLUMN = "customerID"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Valid category values (used for input validation and the Streamlit form)
CATEGORY_VALUES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

# Risk categories (probability of churn)
RISK_LOW_MAX = 0.30
RISK_MEDIUM_MAX = 0.60

# PSI interpretation (common rule of thumb)
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
PSI_BINS = 10

# Artifact file names
CURRENT_MODEL_FILE = "current.json"
EXPERIMENTS_FILE = "experiments.json"
METRICS_FILE = "metrics.json"
REFERENCE_STATS_FILE = "reference_stats.json"
ROC_DATA_FILE = "roc_curves.json"
FEATURE_IMPORTANCE_FILE = "feature_importance.json"
DATASET_INFO_FILE = "dataset_info.json"
PREDICTION_LOG_FILE = "prediction_log.csv"


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR, SCREENSHOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root 'churnsense' logger (console + logs/churnsense.log). Idempotent."""
    ensure_dirs()
    logger = logging.getLogger("churnsense")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    fh = logging.FileHandler(LOGS_DIR / "churnsense.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger
