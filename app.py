"""ChurnSense Streamlit dashboard. Run: streamlit run app.py"""
import html
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.data_loader import load_raw_data
from src.experiment_tracker import experiments_dataframe
from src.insights import generate_insights
from src.monitoring import (compute_drift, drift_summary, load_reference_stats,
                            prediction_stats)
from src.prediction import (ModelNotFoundError, get_current_model_info, load_metadata,
                            load_model, predict_batch, predict_single)

config.setup_logging()

st.set_page_config(page_title="ChurnSense", page_icon="📉", layout="wide")

PALETTE = {"Retained": "#2a9d8f", "Churned": "#e76f51", "primary": "#264653"}
RISK_COLORS = {"Low Risk": "#2a9d8f", "Medium Risk": "#e9c46a", "High Risk": "#e76f51"}

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1250px;}
    div[data-testid="stMetric"] {background: #f6f8fa; border: 1px solid #e3e8ee;
        padding: 14px 16px; border-radius: 10px;}
    .info-card {background: #f6f8fa; border: 1px solid #e3e8ee; padding: 14px 16px;
        border-radius: 10px; min-height: 92px;}
    .info-label {font-size: 0.85rem; opacity: 0.75; margin-bottom: 6px;}
    .info-value {font-size: 1.1rem; font-weight: 600; overflow-wrap: anywhere; line-height: 1.3;}
    @media (prefers-color-scheme: dark) {
      div[data-testid="stMetric"], .info-card {background: #1c2128; border-color: #30363d;}
    }
    h1, h2, h3 {letter-spacing: -0.01em;}
    .small-note {font-size: 0.85rem; opacity: 0.75;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- Cached loaders -------------------------------
@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    df = load_raw_data(auto_download=False)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


@st.cache_resource(show_spinner=False)
def get_model(_version: str):
    return load_model()


def read_json(name: str):
    p = config.ARTIFACTS_DIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def model_or_none():
    info = get_current_model_info()
    if info is None:
        return None
    try:
        return get_model(info["version"])
    except ModelNotFoundError as exc:
        st.error(str(exc))
        return None


def no_model_warning():
    st.warning("No trained model is available yet. Run `python train.py` in a terminal, "
               "or use the **Retrain** button on the *MLOps Monitoring* page.")


# --------------------------------- Pages ------------------------------------
def page_overview():
    st.title("📉 ChurnSense")
    st.caption("Customer churn prediction and retention intelligence · MLOps mini-project")
    try:
        df = get_dataset()
    except FileNotFoundError:
        st.error("Dataset not found. Run `python data/download_data.py` first.")
        return

    churn = (df["Churn"] == "Yes")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{churn.mean() * 100:.1f}%")
    c3.metric("Retained", f"{(~churn).mean() * 100:.1f}%")
    c4.metric("Avg monthly charges", f"${df['MonthlyCharges'].mean():.2f}")

    st.markdown(
        "**About.** ChurnSense trains three scikit-learn classifiers (Logistic Regression, "
        "Random Forest, Gradient Boosting) on the IBM Telco Customer Churn dataset to estimate "
        "the probability that a customer will leave. Alongside prediction it demonstrates "
        "lightweight MLOps practices: model versioning, experiment tracking, drift monitoring, "
        "logging and automated tests. Predictions are statistical estimates, not guaranteed outcomes."
    )

    a, b = st.columns(2)
    with a:
        counts = df["Churn"].map({"No": "Retained", "Yes": "Churned"}).value_counts().reset_index()
        counts.columns = ["Status", "Customers"]
        fig = px.pie(counts, names="Status", values="Customers", hole=0.55, color="Status",
                     color_discrete_map=PALETTE, title="Churn distribution")
        st.plotly_chart(fig, width="stretch")
    with b:
        ct = df.groupby(["Contract", "Churn"]).size().reset_index(name="Customers")
        fig = px.bar(ct, x="Contract", y="Customers", color="Churn", barmode="group",
                     color_discrete_map={"No": PALETTE["Retained"], "Yes": PALETTE["Churned"]},
                     title="Customers by contract type")
        st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        fig = px.histogram(df, x="MonthlyCharges", color="Churn", nbins=40, barmode="overlay",
                           opacity=0.7, color_discrete_map={"No": PALETTE["Retained"], "Yes": PALETTE["Churned"]},
                           title="Monthly charges distribution")
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = px.histogram(df, x="tenure", color="Churn", nbins=36, barmode="overlay", opacity=0.7,
                           color_discrete_map={"No": PALETTE["Retained"], "Yes": PALETTE["Churned"]},
                           title="Customer tenure distribution (months)")
        st.plotly_chart(fig, width="stretch")


def gauge(prob: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%"},
        title={"text": "Churn probability"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": PALETTE["primary"]},
               "steps": [{"range": [0, config.RISK_LOW_MAX * 100], "color": "#bfe6df"},
                         {"range": [config.RISK_LOW_MAX * 100, config.RISK_MEDIUM_MAX * 100], "color": "#f6e3a8"},
                         {"range": [config.RISK_MEDIUM_MAX * 100, 100], "color": "#f4b9a8"}]}))
    fig.update_layout(height=280, margin=dict(t=60, b=10, l=20, r=20))
    return fig


def page_predict():
    st.title("Customer Churn Prediction")
    model = model_or_none()
    if model is None:
        no_model_warning()
        return
    V = config.CATEGORY_VALUES
    with st.form("customer_form"):
        st.subheader("Customer profile")
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", V["gender"])
        senior = c2.selectbox("Senior citizen", V["SeniorCitizen"])
        partner = c3.selectbox("Partner", V["Partner"])
        dependents = c1.selectbox("Dependents", V["Dependents"])
        tenure = c2.number_input("Tenure (months)", 0, 120, 12)
        contract = c3.selectbox("Contract", V["Contract"])

        st.subheader("Services")
        c1, c2, c3 = st.columns(3)
        phone = c1.selectbox("Phone service", V["PhoneService"], index=1)
        multi = c2.selectbox("Multiple lines", ["No", "Yes"])
        internet = c3.selectbox("Internet service", V["InternetService"], index=1)
        osec = c1.selectbox("Online security", ["No", "Yes"])
        obk = c2.selectbox("Online backup", ["No", "Yes"])
        dprot = c3.selectbox("Device protection", ["No", "Yes"])
        tsup = c1.selectbox("Tech support", ["No", "Yes"])
        stv = c2.selectbox("Streaming TV", ["No", "Yes"])
        smov = c3.selectbox("Streaming movies", ["No", "Yes"])
        st.caption("Add-on answers are set to 'No internet service' / 'No phone service' "
                   "automatically when the base service is 'No'.")

        st.subheader("Billing")
        c1, c2, c3 = st.columns(3)
        paperless = c1.selectbox("Paperless billing", V["PaperlessBilling"], index=1)
        payment = c2.selectbox("Payment method", V["PaymentMethod"])
        monthly = c3.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=0.5)
        total = c1.number_input("Total charges ($)", 0.0, 20000.0, float(round(70.0 * 12, 2)), step=10.0)
        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if not submitted:
        return
    no_net = internet == "No"
    cust = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone,
        "MultipleLines": "No phone service" if phone == "No" else multi,
        "InternetService": internet,
        "OnlineSecurity": "No internet service" if no_net else osec,
        "OnlineBackup": "No internet service" if no_net else obk,
        "DeviceProtection": "No internet service" if no_net else dprot,
        "TechSupport": "No internet service" if no_net else tsup,
        "StreamingTV": "No internet service" if no_net else stv,
        "StreamingMovies": "No internet service" if no_net else smov,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }
    try:
        res = predict_single(model, cust)
    except ValueError as exc:
        st.error(str(exc))
        return

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(gauge(res["churn_probability"]), width="stretch")
    with right:
        m1, m2 = st.columns(2)
        m1.metric("Churn probability", f"{res['churn_probability'] * 100:.1f}%")
        m2.metric("Retention probability", f"{res['retention_probability'] * 100:.1f}%")
        for col, label, value in ((m1, "Predicted status", res["predicted_status"]),
                                  (m2, "Risk category", res["risk_category"])):
            col.markdown(f"<div class='info-card'><div class='info-label'>{html.escape(label)}</div>"
                         f"<div class='info-value'>{html.escape(value)}</div></div>",
                         unsafe_allow_html=True)
        st.caption(f"Risk bands: Low < {config.RISK_LOW_MAX:.0%}, Medium {config.RISK_LOW_MAX:.0%}–"
                   f"{config.RISK_MEDIUM_MAX:.0%}, High ≥ {config.RISK_MEDIUM_MAX:.0%}.")
    st.info("This is a model-estimated probability based on historical patterns, not a guaranteed "
            "real-world outcome.")

    st.subheader("Retention insights")
    st.caption("Simple rule-based heuristics, not causal explanations. Treat suggestions as general "
               "business ideas, not guaranteed interventions.")
    for ins in generate_insights(cust, res["risk_category"]):
        with st.expander(f"{ins['factor']}", expanded=True):
            st.write(ins["observation"])
            st.write("💡 " + ins["suggestion"])


def sample_template() -> pd.DataFrame:
    try:
        df = get_dataset()
        return df.drop(columns=["Churn"]).head(10)
    except FileNotFoundError:
        return pd.DataFrame(columns=["customerID"] + config.FEATURE_COLUMNS)


def page_batch():
    st.title("Batch Prediction")
    model = model_or_none()
    if model is None:
        no_model_warning()
        return
    st.write("Upload a CSV with the customer columns used by the model "
             "(`customerID` is optional; a `Churn` column is ignored).")
    st.download_button("Download sample template CSV", sample_template().to_csv(index=False),
                       "churnsense_template.csv", "text/csv")
    up = st.file_uploader("Customer CSV", type=["csv"])
    if up is None:
        return
    try:
        df = pd.read_csv(up)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        return
    try:
        out, report = predict_batch(model, df)
    except ValueError as exc:
        st.error(f"Validation failed: {exc}")
        return

    for w in report["warnings"]:
        st.warning(w)
    if report["invalid_rows"]:
        st.warning(f"{report['invalid_rows']} invalid row(s) were skipped.")
    st.success(f"Scored {len(out)} of {report['n_rows']} rows.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers scored", len(out))
    c2.metric("High-risk customers", int((out["risk_category"] == "High Risk").sum()))
    c3.metric("Mean churn probability", f"{out['churn_probability'].mean() * 100:.1f}%")
    fig = px.histogram(out, x="churn_probability", color="risk_category", nbins=20,
                       color_discrete_map=RISK_COLORS, title="Churn probability distribution")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(out, width="stretch", height=360)
    st.download_button("Download predictions CSV", out.to_csv(index=False),
                       "churnsense_predictions.csv", "text/csv", type="primary")

    ref = load_reference_stats()
    if ref is not None:
        drift = compute_drift(df[report["valid_mask"]], ref)
        st.session_state["last_drift"] = {"table": drift, "rows": len(out), "file": up.name}
        summ = drift_summary(drift)
        if summ["overall"] == "Significant drift":
            st.error("⚠️ " + "  \n".join(summ["warnings"]))
        elif summ["overall"] == "Moderate drift":
            st.warning("  \n".join(summ["warnings"]))
        else:
            st.success("No notable drift versus the training data (PSI).")
        st.caption("Full details are on the MLOps Monitoring page.")


def page_performance():
    st.title("Model Performance")
    metrics = read_json(config.METRICS_FILE)
    if metrics is None:
        no_model_warning()
        return
    selected = metrics["selected_model"]
    st.success(f"Selected model: **{selected}** — chosen by highest cross-validated ROC-AUC "
               f"(version {metrics['model_version']}). Test-set figures below are for reporting only.")
    rows = []
    for name, r in metrics["models"].items():
        t = r["test_metrics"]
        rows.append({"Model": ("★ " if name == selected else "") + name,
                     "CV ROC-AUC": r["cv_roc_auc"], "Accuracy": t["accuracy"],
                     "Precision": t["precision"], "Recall": t["recall"], "F1": t["f1"],
                     "Test ROC-AUC": t["roc_auc"]})
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({c: "{:.4f}" for c in df.columns if c != "Model"}),
                 width="stretch", hide_index=True)
    melted = df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1", "Test ROC-AUC"],
                     var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(melted, x="Metric", y="Score", color="Model", barmode="group",
                           title="Model comparison (held-out test set)", range_y=[0, 1]),
                    width="stretch")

    a, b = st.columns(2)
    with a:
        cm = metrics["models"][selected]["test_metrics"]["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Retained", "Churned"], y=["Retained", "Churned"],
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        title=f"Confusion matrix — {selected}")
        st.plotly_chart(fig, width="stretch")
    with b:
        roc = read_json(config.ROC_DATA_FILE)
        if roc:
            fig = go.Figure()
            for name, d in roc.items():
                auc = metrics["models"][name]["test_metrics"]["roc_auc"]
                fig.add_trace(go.Scatter(x=d["fpr"], y=d["tpr"], mode="lines",
                                         name=f"{name} (AUC {auc:.3f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                     line=dict(dash="dash", color="grey")))
            fig.update_layout(title="ROC curves (test set)", xaxis_title="False positive rate",
                              yaxis_title="True positive rate")
            st.plotly_chart(fig, width="stretch")

    fi = read_json(config.FEATURE_IMPORTANCE_FILE)
    if fi:
        choice = st.selectbox("Feature importance for model", list(fi), index=list(fi).index(selected))
        fdf = pd.DataFrame(fi[choice]).sort_values("importance")
        st.plotly_chart(px.bar(fdf, x="importance", y="feature", orientation="h",
                               title=f"Top features — {choice}", height=560
                               ).update_yaxes(dtick=1), width="stretch")
        st.caption("Importance reflects how much the model relies on a feature (association), "
                   "not a causal effect on churn.")

    with st.expander("Classification report (selected model)"):
        rep = metrics["models"][selected]["test_metrics"]["classification_report"]
        st.dataframe(pd.DataFrame(rep).T.round(4), width="stretch")


def page_monitoring():
    st.title("MLOps Monitoring")
    st.caption("Basic academic monitoring — not production-grade (no streaming, alerting or "
               "label-feedback tracking).")
    info = get_current_model_info()
    meta = load_metadata() if info else None
    warnings_list = []

    if not info or not meta:
        st.warning("No model metadata found. Train a model to populate this page.")
        warnings_list.append("No trained model available.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        # Long text values are shown in info cards because st.metric truncates them
        for col, label, value in (
            (c1, "Current model version", meta["model_version"]),
            (c2, "Algorithm", meta["selected_algorithm"]),
            (c3, "Latest training run", meta["training_date"].replace("T", " ")),
        ):
            col.markdown(f"<div class='info-card'><div class='info-label'>{html.escape(label)}</div>"
                         f"<div class='info-value'>{html.escape(str(value))}</div></div>",
                         unsafe_allow_html=True)
        c4.metric("Test ROC-AUC", f"{meta['test_metrics']['roc_auc']:.4f}")
        with st.expander("Evaluation summary & dataset info", expanded=False):
            t = meta["test_metrics"]
            st.write({k: round(t[k], 4) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")})
            st.json({"hyperparameters": meta["hyperparameters"], "dataset": {
                k: v for k, v in meta["dataset"].items() if k != "cleaning_report"},
                "environment": meta.get("environment")})

    st.subheader("Experiment history")
    exp = experiments_dataframe()
    if exp.empty:
        st.info("No experiments recorded yet.")
    else:
        st.dataframe(exp.sort_values("timestamp", ascending=False), width="stretch",
                     hide_index=True)

    st.subheader("Prediction statistics")
    ps = prediction_stats()
    if ps["count"] == 0:
        st.info("No predictions logged yet. Make a single or batch prediction first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Predictions made", ps["count"])
        c2.metric("High-risk predictions", ps["high_risk"])
        c3.metric("Mean churn probability", f"{ps['mean_probability'] * 100:.1f}%")
        h = ps["histogram"]
        mids = [(h["edges"][i] + h["edges"][i + 1]) / 2 for i in range(len(h["counts"]))]
        st.plotly_chart(px.bar(x=mids, y=h["counts"], labels={"x": "Churn probability", "y": "Predictions"},
                               title="Logged churn-probability distribution"), width="stretch")

    st.subheader("Data drift report (PSI)")
    last = st.session_state.get("last_drift")
    if last is None:
        st.info("Upload a file on the *Batch Prediction* page to compare it with the training data.")
    else:
        st.write(f"Latest batch: **{last['file']}** ({last['rows']} rows) vs. training reference.")
        table = last["table"]
        summ = drift_summary(table)
        warnings_list += summ["warnings"]
        fig = px.bar(table.sort_values("psi"), x="psi", y="feature", orientation="h", color="status",
                     color_discrete_map={"Stable": "#2a9d8f", "Moderate drift": "#e9c46a",
                                         "Significant drift": "#e76f51"})
        fig.add_vline(x=config.PSI_MODERATE, line_dash="dot")
        fig.add_vline(x=config.PSI_SIGNIFICANT, line_dash="dash")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(table, width="stretch", hide_index=True)
        st.caption("PSI < 0.10 stable · 0.10–0.25 moderate · ≥ 0.25 significant (rule-of-thumb thresholds). "
                   "Small uploads give noisy PSI values.")

    st.subheader("Monitoring warnings")
    if warnings_list:
        for w in warnings_list:
            st.warning(w)
    else:
        st.success("No active warnings.")

    st.subheader("Retraining")
    st.write("Runs the full training pipeline (about a minute). The current model is replaced only "
             "if training completes successfully.")
    if st.button("Retrain model", type="primary"):
        from src.training import run_training
        with st.spinner("Training... this can take 1–2 minutes."):
            try:
                out = run_training()
            except Exception as exc:
                st.error(f"Retraining failed; the previous model is still in use. Details: {exc}")
            else:
                get_model.clear()
                st.success(f"New model {out['metadata']['model_version']} "
                           f"({out['metadata']['selected_algorithm']}) is now current.")
                st.rerun()


PAGES = {
    "Overview": page_overview,
    "Churn Prediction": page_predict,
    "Batch Prediction": page_batch,
    "Model Performance": page_performance,
    "MLOps Monitoring": page_monitoring,
}

with st.sidebar:
    st.markdown("## 📉 ChurnSense")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    info = get_current_model_info()
    st.divider()
    if info:
        st.caption(f"Model: {info['selected_algorithm']}\n\nVersion: {info['version']}")
    else:
        st.caption("No model trained yet")
    st.caption("Data: IBM Telco Customer Churn")

PAGES[page]()
