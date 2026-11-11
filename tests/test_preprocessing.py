import numpy as np
from sklearn.linear_model import LogisticRegression

from src import config
from src.data_loader import clean_data
from src.preprocessing import build_pipeline, build_preprocessor, get_feature_names, split_data


def test_split_is_stratified_80_20_and_reproducible(raw_df):
    X, y, _ = clean_data(raw_df)
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    assert abs(len(X_te) / len(X) - 0.2) < 0.01
    assert abs(y_tr.mean() - y_te.mean()) < 0.01
    X_tr2, X_te2, _, _ = split_data(X, y)
    assert X_te.index.equals(X_te2.index)


def test_preprocessor_outputs_finite_numeric_matrix(raw_df):
    X, y, _ = clean_data(raw_df)
    X_tr, X_te, _, _ = split_data(X, y)
    pre = build_preprocessor().fit(X_tr)
    out = pre.transform(X_te)
    assert out.shape[0] == len(X_te)
    assert np.isfinite(out).all()


def test_scaler_fit_on_train_only(raw_df):
    X, y, _ = clean_data(raw_df)
    X_tr, X_te, _, _ = split_data(X, y)
    pre = build_preprocessor().fit(X_tr)
    scaler = pre.named_transformers_["num"].named_steps["scaler"]
    assert np.isclose(scaler.mean_[0], X_tr["tenure"].mean(), atol=1e-6)


def test_unseen_category_does_not_crash(raw_df):
    X, y, _ = clean_data(raw_df)
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    row = X.head(1).copy()
    row["Contract"] = "Three year"  # unseen
    assert pipe.predict_proba(row).shape == (1, 2)


def test_feature_names_match_matrix_width(raw_df):
    X, y, _ = clean_data(raw_df)
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    names = get_feature_names(pipe)
    assert len(names) == pipe.named_steps["model"].coef_.shape[1]
    assert set(config.NUMERIC_FEATURES) <= set(names)
