"""Training page: run the ML pipeline and display metrics."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIError
from dashboard.i18n import t
from dashboard.theme import (
    apply_page_config,
    get_api_client,
    render_settings_sidebar,
    require_dataset_id,
    show_error,
)

apply_page_config()
render_settings_sidebar()

st.title(t("training_title"))
st.caption(t("training_caption"))

dataset_id = require_dataset_id()
if dataset_id is None:
    st.stop()

client = get_api_client()

with st.spinner(t("loading_columns")):
    try:
        summary = client.summary(dataset_id)
    except APIError as exc:
        st.error(f"{t('api_error')} ({exc.status_code}): {exc.message}")
        st.stop()
    except Exception as exc:
        show_error(exc)
        st.stop()

columns = list((summary.get("dtypes") or {}).keys())

with st.form("training-form"):
    target = st.selectbox(t("target_column"), options=columns)
    problem_type = st.selectbox(t("problem_type"), ["classification", "regression"])
    col_test, col_cv = st.columns(2)
    test_size = col_test.slider(t("test_size"), 0.05, 0.5, 0.2, 0.05)
    cv_folds = col_cv.slider(t("cv_folds"), 2, 10, 5)
    submitted = st.form_submit_button(t("train_button"))

if submitted and target:
    body = {
        "target_column": target,
        "problem_type": problem_type,
        "test_size": test_size,
        "cv_folds": cv_folds,
    }
    with st.spinner(t("training_running")):
        try:
            result = client.train(dataset_id, body)
        except APIError as exc:
            st.error(f"{t('api_error')} ({exc.status_code}): {exc.message}")
            st.stop()
        except Exception as exc:
            show_error(exc)
            st.stop()

    st.session_state["model_id"] = result.get("model_id")
    st.success(f"{t('winner')}: **{result.get('chosen_algorithm')}**")

    col1, col2, col3 = st.columns(3)
    col1.metric(t("samples_train"), result.get("n_samples_train", "?"))
    col2.metric(t("samples_test"), result.get("n_samples_test", "?"))
    col3.metric(
        t("model_id"),
        result.get("model_id", "?")[:8] + "..." if result.get("model_id") else "?",
    )

    st.markdown(f"**{t('candidate_metrics')}**")
    candidates = result.get("candidates") or []
    if candidates:
        rows = []
        for candidate in candidates:
            metrics = candidate.get("test_metrics", {}) or {}
            row = {
                "algorithm": candidate.get("algorithm"),
                "cv_score_mean": round(candidate.get("cv_score_mean", 0), 4),
                "cv_score_std": round(candidate.get("cv_score_std", 0), 4),
            }
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "r2", "mae", "rmse"):
                value = metrics.get(key)
                if value is not None:
                    row[key] = round(value, 4)
            rows.append(row)
        st.dataframe(rows, hide_index=True, use_container_width=True)

    st.caption(f"{t('model_saved_at')} `{result.get('model_uri', '?')}`")
