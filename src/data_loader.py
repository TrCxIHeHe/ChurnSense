"""Data loading and cleaning (no model code here)."""
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger("churnsense.data")

REQUIRED_RAW_COLUMNS = [config.ID_COLUMN] + config.FEATURE_COLUMNS + [config.TARGET]


def download_dataset(dest: Path = config.DATA_FILE, url: str = config.DATA_SOURCE_URL) -> Path:
    """Download the real IBM Telco churn CSV. Raises on failure (never substitutes fake data)."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset from %s", url)
    urllib.request.urlretrieve(url, dest)  # nosec - fixed public URL
    logger.info("Saved dataset to %s", dest)
    return dest


def load_raw_data(path: Optional[Path] = None, auto_download: bool = True) -> pd.DataFrame:
    """Load the raw churn CSV. Tries a download if the file is missing."""
    path = Path(path) if path else config.DATA_FILE
    if not path.exists():
        if auto_download:
            try:
                download_dataset(path)
            except Exception as exc:  # network unavailable etc.
                logger.error("Dataset download failed: %s", exc)
                raise FileNotFoundError(
                    f"Dataset not found at {path} and download failed ({exc}). "
                    f"Download it manually from {config.DATA_SOURCE_URL} and place it at {path}."
                ) from exc
        else:
            raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    logger.info("Loaded raw data: %d rows x %d columns from %s", *df.shape, path)
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Type correction and value normalisation for the feature columns.

    Shared by training and inference so both see identical representations.
    Does not drop rows. Unparseable numerics become NaN (imputed later).
    """
    out = df.copy()
    for col in config.CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
    if "SeniorCitizen" in out.columns:
        mapping = {"0": "No", "1": "Yes", "0.0": "No", "1.0": "Yes",
                   "no": "No", "yes": "Yes", "NO": "No", "YES": "Yes",
                   "false": "No", "true": "Yes", "False": "No", "True": "Yes"}
        out["SeniorCitizen"] = out["SeniorCitizen"].replace(mapping)
    for col in config.NUMERIC_FEATURES:
        if col in out.columns:
            # plain float64 (NaN for blanks/garbage), never a nullable dtype with pd.NA
            out[col] = pd.to_numeric(out[col].astype("string").str.strip(),
                                     errors="coerce").astype("float64")
    for col in config.CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype(object)
    return out


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Clean the raw training data. Returns (X, y, report)."""
    report = {"rows_raw": int(len(df))}
    df = df.copy()

    # Missing values (incl. blank strings in TotalCharges)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = normalize_features(df)
    report["missing_before"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}

    # Duplicates (ignoring the unique customer ID)
    dup_mask = df.drop(columns=[config.ID_COLUMN]).duplicated()
    report["duplicate_rows"] = int(dup_mask.sum())
    if report["duplicate_rows"]:
        logger.warning("Dropping %d duplicate records", report["duplicate_rows"])
        df = df[~dup_mask]

    # Rows without a target label cannot be used for supervised learning
    no_target = df[config.TARGET].isna()
    report["rows_without_target_dropped"] = int(no_target.sum())
    df = df[~no_target]

    # TotalCharges is blank for tenure == 0 customers -> logically 0; otherwise median
    tc_missing = df["TotalCharges"].isna()
    report["total_charges_missing"] = int(tc_missing.sum())
    df.loc[tc_missing & (df["tenure"] == 0), "TotalCharges"] = 0.0
    # Remaining numeric / categorical gaps are left for the pipeline imputers (no leakage)

    y = df[config.TARGET].astype(str).str.strip().map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Target column 'Churn' contains values other than Yes/No")
    X = df[config.FEATURE_COLUMNS].reset_index(drop=True)
    y = y.astype(int).reset_index(drop=True)

    report["rows_clean"] = int(len(X))
    report["churn_rate"] = float(y.mean())
    report["identifier_removed"] = config.ID_COLUMN
    logger.info("Cleaned data: %d rows, churn rate %.2f%%", len(X), 100 * y.mean())
    return X, y, report


def dataset_info(df_raw: pd.DataFrame, report: dict, path: Optional[Path] = None) -> dict:
    return {
        "name": config.DATA_SOURCE_NAME,
        "source_url": config.DATA_SOURCE_URL,
        "source_page": config.DATA_SOURCE_PAGE,
        "local_path": str(Path(path) if path else config.DATA_FILE),
        "is_synthetic": False,
        "rows_raw": int(len(df_raw)),
        "columns_raw": int(df_raw.shape[1]),
        "cleaning_report": report,
    }
