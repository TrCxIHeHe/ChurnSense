"""Reproducible training pipeline: tuning, selection, versioning, tracking."""
import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import sklearn
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src import config, data_loader, evaluation, experiment_tracker, monitoring
from src.preprocessing import build_pipeline, split_data

logger = logging.getLogger("churnsense.train")


def get_model_specs() -> dict:
    """Estimators and RandomizedSearchCV search spaces (kept small for speed)."""
    rs = config.RANDOM_STATE
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, random_state=rs),
            {"model__C": loguniform(1e-2, 1e2),
             "model__class_weight": [None, "balanced"]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=rs, n_jobs=-1),
            {"model__n_estimators": randint(100, 400),
             "model__max_depth": [4, 6, 8, 12, None],
             "model__min_samples_leaf": randint(1, 12),
             "model__max_features": ["sqrt", "log2"],
             "model__class_weight": [None, "balanced"]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=rs),
            {"model__n_estimators": randint(80, 300),
             "model__learning_rate": loguniform(0.01, 0.3),
             "model__max_depth": randint(2, 5),
             "model__subsample": uniform(0.6, 0.4),
             "model__min_samples_leaf": randint(1, 12)},
        ),
    }


def _clean_params(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        k = k.replace("model__", "")
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        out[k] = v
    return out


def run_training(data_path: Optional[Path] = None, models_dir: Optional[Path] = None,
                 artifacts_dir: Optional[Path] = None, n_iter: Optional[int] = None,
                 cv_folds: Optional[int] = None, save_plots: bool = True) -> dict:
    """Run the full pipeline. The 'current' model pointer is only updated on success,
    so a failed retraining never breaks the previously deployed model."""
    models_dir = Path(models_dir or config.MODELS_DIR)
    artifacts_dir = Path(artifacts_dir or config.ARTIFACTS_DIR)
    n_iter = n_iter or config.N_ITER_SEARCH
    cv_folds = cv_folds or config.CV_FOLDS
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"v{run_id}"
    logger.info("=== Training run %s started (n_iter=%d, cv=%d) ===", run_id, n_iter, cv_folds)

    # 1. Data
    raw = data_loader.load_raw_data(data_path)
    X, y, report = data_loader.clean_data(raw)
    ds_info = data_loader.dataset_info(raw, report, data_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    ds_info.update({"train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                    "split": f"{int((1 - config.TEST_SIZE) * 100)}/{int(config.TEST_SIZE * 100)} stratified, "
                             f"random_state={config.RANDOM_STATE}"})
    logger.info("Split: train=%d test=%d", len(X_train), len(X_test))

    # 2. Tune + cross-validate each model (CV happens on the TRAINING set only)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    results, fitted = {}, {}
    for name, (est, space) in get_model_specs().items():
        logger.info("Tuning %s ...", name)
        search = RandomizedSearchCV(
            build_pipeline(est), space, n_iter=n_iter, scoring="roc_auc", cv=cv,
            random_state=config.RANDOM_STATE, n_jobs=-1, refit=True,
        )
        search.fit(X_train, y_train)
        logger.info("%s best CV ROC-AUC = %.4f params=%s", name, search.best_score_,
                    _clean_params(search.best_params_))
        fitted[name] = search.best_estimator_
        results[name] = {
            "cv_roc_auc": float(search.best_score_),
            "cv_roc_auc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "best_params": _clean_params(search.best_params_),
        }

    # 3. Select by cross-validated (validation) ROC-AUC -- NOT by test results
    selected = max(results, key=lambda n: results[n]["cv_roc_auc"])
    logger.info("Selected model by CV ROC-AUC: %s (%.4f)", selected, results[selected]["cv_roc_auc"])

    # 4. Evaluate every model once on the untouched test set (reporting only)
    roc, aucs = {}, {}
    for name, pipe in fitted.items():
        results[name]["test_metrics"] = evaluation.evaluate_model(pipe, X_test, y_test)
        roc[name] = evaluation.roc_data(pipe, X_test, y_test)
        aucs[name] = results[name]["test_metrics"]["roc_auc"]
        results[name]["feature_importance"] = evaluation.feature_importance(fitted[name])

    # 5. Persist model (versioned), metadata, artifacts
    model_file = f"churnsense_{version}.joblib"
    meta_file = f"churnsense_{version}_metadata.json"
    joblib.dump(fitted[selected], models_dir / model_file)

    metadata = {
        "model_version": version,
        "run_id": run_id,
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "selected_algorithm": selected,
        "selection_criterion": "highest mean cross-validated ROC-AUC on the training set",
        "hyperparameters": results[selected]["best_params"],
        "cv_roc_auc": results[selected]["cv_roc_auc"],
        "test_metrics": results[selected]["test_metrics"],
        "dataset": ds_info,
        "random_state": config.RANDOM_STATE,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }
    (models_dir / meta_file).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    all_metrics = {
        "run_id": run_id, "model_version": version, "selected_model": selected,
        "models": {n: {k: v for k, v in r.items() if k != "feature_importance"} for n, r in results.items()},
    }
    (artifacts_dir / config.METRICS_FILE).write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    (artifacts_dir / config.ROC_DATA_FILE).write_text(json.dumps(roc), encoding="utf-8")
    (artifacts_dir / config.FEATURE_IMPORTANCE_FILE).write_text(
        json.dumps({n: r["feature_importance"] for n, r in results.items()}, indent=2), encoding="utf-8")
    (artifacts_dir / config.DATASET_INFO_FILE).write_text(json.dumps(ds_info, indent=2), encoding="utf-8")
    monitoring.save_reference_stats(monitoring.build_reference_stats(X_train), artifacts_dir)

    # 6. Experiment tracking (one record per model)
    for name, r in results.items():
        experiment_tracker.log_experiment(
            run_id, name, r["best_params"],
            {k: r["test_metrics"][k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
            {"name": ds_info["name"], "rows": ds_info["rows_raw"], "train_rows": ds_info["train_rows"],
             "test_rows": ds_info["test_rows"]},
            selected=(name == selected), cv_roc_auc=r["cv_roc_auc"], model_version=version,
            artifacts_dir=artifacts_dir)
    experiment_tracker.export_csv(artifacts_dir)

    # 7. Plots
    if save_plots:
        try:
            evaluation.plot_churn_distribution(y, artifacts_dir / "churn_distribution.png")
            evaluation.plot_model_comparison(results, artifacts_dir / "model_comparison.png")
            evaluation.plot_confusion_matrix(results[selected]["test_metrics"]["confusion_matrix"],
                                             f"Confusion Matrix - {selected}",
                                             artifacts_dir / "confusion_matrix.png")
            evaluation.plot_roc_curves(roc, aucs, artifacts_dir / "roc_curves.png")
            evaluation.plot_feature_importance(results[selected]["feature_importance"],
                                               f"Top Features - {selected}",
                                               artifacts_dir / "feature_importance.png")
        except Exception as exc:  # plots must never break training
            logger.warning("Plot generation failed: %s", exc)

    # 8. Atomically switch the current-model reference (last step = safe retraining)
    pointer = {"version": version, "model_file": model_file, "metadata_file": meta_file,
               "selected_algorithm": selected, "updated": datetime.now().isoformat(timespec="seconds")}
    tmp = models_dir / (config.CURRENT_MODEL_FILE + ".tmp")
    tmp.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    tmp.replace(models_dir / config.CURRENT_MODEL_FILE)
    logger.info("=== Run %s finished. Current model -> %s (%s) ===", run_id, version, selected)

    return {"metadata": metadata, "results": all_metrics, "pointer": pointer}
