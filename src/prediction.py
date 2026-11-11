"""Model loading and inference (single + batch)."""
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data_loader import normalize_features
from src.monitoring import log_predictions, validate_input

logger = logging.getLogger("churnsense.predict")


class ModelNotFoundError(RuntimeError):
    pass


def risk_category(p: float) -> str:
    if p < config.RISK_LOW_MAX:
        return "Low Risk"
    if p < config.RISK_MEDIUM_MAX:
        return "Medium Risk"
    return "High Risk"


def get_current_model_info(models_dir: Optional[Path] = None) -> Optional[dict]:
    """Return the 'current model' reference record, or None."""
    p = Path(models_dir or config.MODELS_DIR) / config.CURRENT_MODEL_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.error("Corrupted %s", p)
        return None


def load_model(models_dir: Optional[Path] = None):
    """Load the current model pipeline. Raises ModelNotFoundError if unavailable."""
    models_dir = Path(models_dir or config.MODELS_DIR)
    info = get_current_model_info(models_dir)
    if info is None:
        raise ModelNotFoundError("No trained model found. Run `python train.py` first.")
    path = models_dir / info["model_file"]
    if not path.exists():
        raise ModelNotFoundError(f"Model file {path.name} referenced by {config.CURRENT_MODEL_FILE} is missing.")
    # Pickled scikit-learn models are only guaranteed to load on the version that saved them.
    try:
        import sklearn
        meta = load_metadata(models_dir) or {}
        saved = (meta.get("environment") or {}).get("scikit_learn")
        if saved and saved != sklearn.__version__:
            logger.warning("Model was trained with scikit-learn %s but %s is installed; "
                           "if loading or predictions fail, run `python train.py`.",
                           saved, sklearn.__version__)
    except Exception:  # version check must never block loading
        pass
    try:
        model = joblib.load(path)
    except Exception as exc:
        logger.error("Could not load %s: %s", path.name, exc)
        raise ModelNotFoundError(
            f"Model file {path.name} could not be loaded ({type(exc).__name__}). This usually means a "
            "different scikit-learn version than the one used for training. Run `python train.py` "
            "to retrain with your installed versions.") from exc
    logger.info("Loaded model %s from %s", info.get("version"), path.name)
    return model


def load_metadata(models_dir: Optional[Path] = None) -> Optional[dict]:
    models_dir = Path(models_dir or config.MODELS_DIR)
    info = get_current_model_info(models_dir)
    if not info:
        return None
    p = models_dir / info["metadata_file"]
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_features(df)[config.FEATURE_COLUMNS]


def predict_single(model, customer: dict, artifacts_dir: Optional[Path] = None,
                   log: bool = True) -> dict:
    """Predict churn for one customer dict. Returns probabilities, label and risk."""
    df = pd.DataFrame([customer])
    report = validate_input(df)
    if report["errors"] or not report["valid_mask"].all():
        raise ValueError("Invalid customer input: " + "; ".join(report["errors"] + report["warnings"]))
    prob = float(model.predict_proba(_prepare(df))[0, 1])
    result = {
        "churn_probability": prob,
        "retention_probability": 1.0 - prob,
        "predicted_churn": bool(prob >= 0.5),
        "predicted_status": "Likely to churn" if prob >= 0.5 else "Likely to stay",
        "risk_category": risk_category(prob),
    }
    logger.info("Single prediction: p(churn)=%.3f risk=%s", prob, result["risk_category"])
    if log:
        log_predictions([prob], "single", artifacts_dir)
    return result


def predict_batch(model, df: pd.DataFrame, artifacts_dir: Optional[Path] = None,
                  log: bool = True):
    """Predict for a DataFrame. Returns (results_df, validation_report).

    Rows failing validation are excluded from scoring (and counted in the report).
    """
    report = validate_input(df)
    if report["errors"]:
        raise ValueError("; ".join(report["errors"]))
    valid = report["valid_mask"]
    scored = df[valid].copy()
    if scored.empty:
        raise ValueError("No valid rows to score.")
    probs = model.predict_proba(_prepare(scored))[:, 1]
    out = scored.copy()
    out["churn_probability"] = np.round(probs, 4)
    out["retention_probability"] = np.round(1 - probs, 4)
    out["predicted_churn"] = np.where(probs >= 0.5, "Yes", "No")
    out["risk_category"] = [risk_category(p) for p in probs]
    logger.info("Batch prediction: %d scored, %d skipped as invalid", len(out), int((~valid).sum()))
    if log:
        log_predictions(probs, "batch", artifacts_dir)
    return out, report
