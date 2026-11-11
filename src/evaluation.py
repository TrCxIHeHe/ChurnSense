"""Metrics and plots."""
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from src.preprocessing import get_feature_names  # noqa: E402

logger = logging.getLogger("churnsense.eval")


def evaluate_model(pipeline, X_test, y_test) -> dict:
    """Evaluate a fitted pipeline on held-out data. Threshold = 0.5."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, pred, target_names=["Retained", "Churned"], output_dict=True, zero_division=0
        ),
    }
    logger.info("Eval: acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f",
                metrics["accuracy"], metrics["precision"], metrics["recall"],
                metrics["f1"], metrics["roc_auc"])
    return metrics


def roc_data(pipeline, X_test, y_test, max_points: int = 200) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    if len(fpr) > max_points:
        idx = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


def feature_importance(pipeline, top_n: int = 20) -> list:
    """Model-native importance: |coef| for LR, impurity importance for tree ensembles.

    NOTE: importance indicates association used by the model, not causation.
    """
    model = pipeline.named_steps["model"]
    names = get_feature_names(pipeline)
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        vals = np.abs(np.ravel(model.coef_))
    else:
        return []
    df = pd.DataFrame({"feature": names, "importance": vals}).sort_values("importance", ascending=False)
    return df.head(top_n).to_dict(orient="records")


# ----------------------------- Plots (PNG) ---------------------------------
def plot_churn_distribution(y, out: Path) -> None:
    counts = pd.Series(y).map({0: "Retained", 1: "Churned"}).value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=["#2a9d8f", "#e76f51"][: len(counts)])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({100 * v / counts.sum():.1f}%)", ha="center", va="bottom")
    ax.set_title("Churn Distribution")
    ax.set_ylabel("Customers")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_model_comparison(results: dict, out: Path) -> None:
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    df = pd.DataFrame({m: {k: r["test_metrics"][k] for k in keys} for m, r in results.items()})
    ax = df.T.plot(kind="bar", figsize=(9, 4.5), rot=0, colormap="viridis")
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison (held-out test set)")
    ax.legend(loc="lower right", ncol=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_confusion_matrix(cm, title: str, out: Path) -> None:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Retained", "Churned"])
    ax.set_yticks([0, 1], ["Retained", "Churned"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_roc_curves(roc: dict, aucs: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, d in roc.items():
        ax.plot(d["fpr"], d["tpr"], label=f"{name} (AUC={aucs[name]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_feature_importance(fi: list, title: str, out: Path) -> None:
    if not fi:
        return
    df = pd.DataFrame(fi).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(df["feature"], df["importance"], color="#264653")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
