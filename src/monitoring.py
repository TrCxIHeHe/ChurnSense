"""Basic academic model monitoring: input validation, PSI drift, prediction stats.

NOTE: This is a simple educational implementation, NOT production-grade monitoring
(no streaming, no alerting, no concept-drift/label-feedback tracking).
"""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src.data_loader import normalize_features

logger = logging.getLogger("churnsense.monitor")


# ------------------------- Input validation --------------------------------
def validate_input(df: pd.DataFrame) -> dict:
    """Validate an uploaded batch. Returns a report; `valid_mask` marks usable rows.

    Rules: all required feature columns present; numeric columns parseable & non-negative;
    categorical values within the known vocabulary (unknown values are flagged as warnings
    because the encoder ignores them, but the row is still scored).
    """
    report = {"errors": [], "warnings": [], "n_rows": int(len(df)), "missing_columns": [],
              "extra_columns": [], "invalid_rows": 0}
    cols = list(df.columns)
    report["missing_columns"] = [c for c in config.FEATURE_COLUMNS if c not in cols]
    report["extra_columns"] = [c for c in cols if c not in config.FEATURE_COLUMNS
                               and c not in (config.ID_COLUMN, config.TARGET)]
    if report["missing_columns"]:
        report["errors"].append(f"Missing required columns: {report['missing_columns']}")
        report["valid_mask"] = pd.Series(False, index=df.index)
        return report
    if len(df) == 0:
        report["errors"].append("Uploaded file contains no rows.")
        report["valid_mask"] = pd.Series(dtype=bool)
        return report
    if report["extra_columns"]:
        report["warnings"].append(f"Ignoring unexpected columns: {report['extra_columns']}")

    norm = normalize_features(df)
    valid = pd.Series(True, index=df.index)
    for col in config.NUMERIC_FEATURES:
        negative = (norm[col] < 0).fillna(False).astype(bool)
        if col == "TotalCharges":
            # Blank TotalCharges is legitimate for brand-new customers -> imputed, row stays valid
            bad = negative
            n_blank = int(norm[col].isna().sum())
            if n_blank:
                report["warnings"].append(f"{n_blank} row(s) have blank 'TotalCharges' (will be imputed)")
        else:
            bad = norm[col].isna() | negative
        if bad.any():
            report["warnings"].append(f"{int(bad.sum())} row(s) have missing/invalid '{col}'")
        valid &= ~bad
    for col in config.CATEGORICAL_FEATURES:
        allowed = set(config.CATEGORY_VALUES[col])
        isna = norm[col].isna()
        unknown = ~norm[col].isin(allowed) & ~isna
        if isna.any():
            report["warnings"].append(f"{int(isna.sum())} row(s) have missing '{col}' (will be imputed)")
        if unknown.any():
            vals = sorted(map(str, norm.loc[unknown, col].unique()))[:5]
            report["warnings"].append(f"{int(unknown.sum())} row(s) have unexpected '{col}' values {vals}")
    report["valid_mask"] = valid
    report["invalid_rows"] = int((~valid).sum())
    return report


# ------------------------------- PSI ---------------------------------------
def psi_from_proportions(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def psi_label(value: float) -> str:
    if value >= config.PSI_SIGNIFICANT:
        return "Significant drift"
    if value >= config.PSI_MODERATE:
        return "Moderate drift"
    return "Stable"


def build_reference_stats(X_train: pd.DataFrame) -> dict:
    """Reference distribution computed from the TRAINING features only."""
    stats = {"numeric": {}, "categorical": {}, "n_rows": int(len(X_train))}
    for col in config.NUMERIC_FEATURES:
        s = pd.to_numeric(X_train[col], errors="coerce").dropna()
        edges = np.unique(np.quantile(s, np.linspace(0, 1, config.PSI_BINS + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(s, bins=edges)
        stats["numeric"][col] = {
            "edges": [None if np.isinf(e) else float(e) for e in edges],
            "proportions": (counts / counts.sum()).tolist(),
            "mean": float(s.mean()), "std": float(s.std()),
        }
    for col in config.CATEGORICAL_FEATURES:
        vc = X_train[col].astype(str).value_counts(normalize=True)
        stats["categorical"][col] = {k: float(v) for k, v in vc.items()}
    return stats


def save_reference_stats(stats: dict, artifacts_dir: Optional[Path] = None) -> Path:
    p = Path(artifacts_dir or config.ARTIFACTS_DIR) / config.REFERENCE_STATS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Saved reference statistics to %s", p)
    return p


def load_reference_stats(artifacts_dir: Optional[Path] = None) -> Optional[dict]:
    p = Path(artifacts_dir or config.ARTIFACTS_DIR) / config.REFERENCE_STATS_FILE
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def compute_drift(df: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """PSI per feature of `df` vs the training reference."""
    norm = normalize_features(df)
    rows = []
    for col, ref in reference["numeric"].items():
        if col not in norm:
            continue
        s = norm[col].dropna()
        if s.empty:
            continue
        edges = np.array([-np.inf if e is None and i == 0 else (np.inf if e is None else e)
                          for i, e in enumerate(ref["edges"])])
        counts, _ = np.histogram(s, bins=edges)
        actual = counts / max(counts.sum(), 1)
        v = psi_from_proportions(np.array(ref["proportions"]), actual)
        rows.append({"feature": col, "type": "numeric", "psi": v, "status": psi_label(v)})
    for col, ref in reference["categorical"].items():
        if col not in norm:
            continue
        cats = sorted(set(ref) | set(norm[col].dropna().astype(str).unique()))
        exp = np.array([ref.get(c, 0.0) for c in cats])
        act_counts = norm[col].dropna().astype(str).value_counts()
        act = np.array([act_counts.get(c, 0) for c in cats], dtype=float)
        if act.sum() == 0:
            continue
        v = psi_from_proportions(exp, act / act.sum())
        rows.append({"feature": col, "type": "categorical", "psi": v, "status": psi_label(v)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def drift_summary(drift_df: pd.DataFrame) -> dict:
    if drift_df.empty:
        return {"overall": "Unknown", "n_significant": 0, "n_moderate": 0, "warnings": []}
    sig = drift_df[drift_df["psi"] >= config.PSI_SIGNIFICANT]
    mod = drift_df[(drift_df["psi"] >= config.PSI_MODERATE) & (drift_df["psi"] < config.PSI_SIGNIFICANT)]
    warnings = []
    if len(sig):
        warnings.append(f"Significant drift (PSI >= {config.PSI_SIGNIFICANT}) in: {', '.join(sig['feature'])}")
    if len(mod):
        warnings.append(f"Moderate drift (PSI >= {config.PSI_MODERATE}) in: {', '.join(mod['feature'])}")
    overall = "Significant drift" if len(sig) else ("Moderate drift" if len(mod) else "Stable")
    return {"overall": overall, "n_significant": int(len(sig)), "n_moderate": int(len(mod)),
            "warnings": warnings}


# -------------------------- Prediction logging -----------------------------
def log_predictions(probabilities, source: str, artifacts_dir: Optional[Path] = None) -> None:
    """Append prediction summary rows (probability + risk) for monitoring statistics."""
    from src.prediction import risk_category

    probs = np.asarray(probabilities, dtype=float)
    p = Path(artifacts_dir or config.ARTIFACTS_DIR) / config.PREDICTION_LOG_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().isoformat(timespec="seconds")
    rows = pd.DataFrame({"timestamp": ts, "source": source, "churn_probability": probs,
                         "risk": [risk_category(x) for x in probs]})
    rows.to_csv(p, mode="a", header=not p.exists(), index=False)


def prediction_stats(artifacts_dir: Optional[Path] = None) -> dict:
    p = Path(artifacts_dir or config.ARTIFACTS_DIR) / config.PREDICTION_LOG_FILE
    if not p.exists():
        return {"count": 0, "high_risk": 0, "mean_probability": None, "histogram": None,
                "by_source": {}}
    df = pd.read_csv(p)
    if df.empty:
        return {"count": 0, "high_risk": 0, "mean_probability": None, "histogram": None,
                "by_source": {}}
    counts, edges = np.histogram(df["churn_probability"], bins=10, range=(0, 1))
    return {
        "count": int(len(df)),
        "high_risk": int((df["risk"] == "High Risk").sum()),
        "mean_probability": float(df["churn_probability"].mean()),
        "histogram": {"counts": counts.tolist(), "edges": edges.tolist()},
        "by_source": df["source"].value_counts().to_dict(),
        "risk_counts": df["risk"].value_counts().to_dict(),
    }
