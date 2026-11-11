"""Lightweight JSON experiment tracker (no external services)."""
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config

logger = logging.getLogger("churnsense.tracker")


def _path(artifacts_dir: Optional[Path]) -> Path:
    return Path(artifacts_dir or config.ARTIFACTS_DIR) / config.EXPERIMENTS_FILE


def load_experiments(artifacts_dir: Optional[Path] = None) -> list:
    p = _path(artifacts_dir)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.error("Experiment file %s is corrupted", p)
        return []


def new_experiment_id() -> str:
    return "exp-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def log_experiment(run_id: str, model_name: str, hyperparameters: dict, metrics: dict,
                   dataset: dict, selected: bool = False, cv_roc_auc: Optional[float] = None,
                   model_version: Optional[str] = None,
                   artifacts_dir: Optional[Path] = None) -> dict:
    """Append one experiment record (one per model per training run)."""
    record = {
        "experiment_id": new_experiment_id(),
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "hyperparameters": {k: (v.item() if hasattr(v, "item") else v) for k, v in hyperparameters.items()},
        "cv_roc_auc": cv_roc_auc,
        "test_metrics": metrics,
        "dataset": dataset,
        "selected": bool(selected),
        "model_version": model_version,
    }
    records = load_experiments(artifacts_dir)
    records.append(record)
    p = _path(artifacts_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    logger.info("Logged experiment %s (%s)", record["experiment_id"], model_name)
    return record


def experiments_dataframe(artifacts_dir: Optional[Path] = None) -> pd.DataFrame:
    rows = []
    for r in load_experiments(artifacts_dir):
        m = r.get("test_metrics", {})
        rows.append({
            "experiment_id": r["experiment_id"],
            "run_id": r["run_id"],
            "timestamp": r["timestamp"],
            "model": r["model_name"],
            "cv_roc_auc": r.get("cv_roc_auc"),
            "accuracy": m.get("accuracy"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1": m.get("f1"),
            "test_roc_auc": m.get("roc_auc"),
            "selected": r.get("selected"),
            "model_version": r.get("model_version"),
        })
    return pd.DataFrame(rows)


def export_csv(artifacts_dir: Optional[Path] = None) -> Path:
    out = Path(artifacts_dir or config.ARTIFACTS_DIR) / "experiments.csv"
    experiments_dataframe(artifacts_dir).to_csv(out, index=False)
    return out
