"""Explanations page: feature importance, SHAP values, summary plot."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIError
from dashboard.theme import apply_page_config, get_api_client, show_error

apply_page_config("Explanations - Dashboard")

st.title("Explicabilidade")
st.caption("Feature importance + SHAP values do modelo treinado.")

client = get_api_client()

default_model = st.session_state.get("model_id", "")
model_id = st.text_input("Model ID", value=default_model)

if not model_id:
    st.info("Treine um modelo em **Training** ou informe um Model ID manualmente.")
    st.stop()

if st.button("Explicar modelo"):
    with st.spinner("Calculando SHAP..."):
        try:
            result = client.explain(model_id)
        except APIError as exc:
            st.error(f"API retornou {exc.status_code}: {exc.message}")
            st.stop()
        except Exception as exc:  # noqa: BLE001
            show_error(exc)
            st.stop()

    st.session_state["last_explanation"] = result

data = st.session_state.get("last_explanation")
if not data:
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Algoritmo", data.get("algorithm", "?"))
col2.metric("Problem type", data.get("problem_type", "?"))
col3.metric("Target", data.get("target_column", "?"))

st.info(data.get("narrative") or "")

st.subheader("Feature importance")
importance = data.get("feature_importance") or []
if importance:
    rows = [
        {"feature": item.get("feature"), "importance": round(item.get("importance", 0), 4)}
        for item in importance
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)

st.subheader("SHAP - mean absolute values")
shap = data.get("shap") or {}
values = shap.get("mean_abs_values") or []
if values:
    rows = [
        {"feature": item.get("feature"), "mean_abs_shap": round(item.get("value", 0), 4)}
        for item in values
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)

chart_url = shap.get("chart_url")
if chart_url:
    st.subheader("SHAP summary plot")
    absolute = chart_url if chart_url.startswith("http") else f"{client.base_url}{chart_url}"
    st.image(absolute, use_container_width=True)
