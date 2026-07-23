"""Insights page: LLM-authored structured analysis."""

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

st.title(t("insights_title"))
st.caption(t("insights_caption"))

dataset_id = require_dataset_id()
if dataset_id is None:
    st.stop()

client = get_api_client()

with st.form("insights-form"):
    col_prov, col_model = st.columns(2)
    provider = col_prov.selectbox(
        t("provider_optional"),
        options=["", "openai", "anthropic", "gemini"],
        format_func=lambda v: v or "default",
    )
    model = col_model.text_input(t("model_optional"))
    submitted = st.form_submit_button(t("generate_insights"))

if submitted:
    body: dict = {}
    if provider:
        body["provider"] = provider
    if model:
        body["model"] = model
    with st.spinner(t("asking_llm")):
        try:
            result = client.insights(dataset_id, body)
        except APIError as exc:
            st.error(f"{t('api_error')} ({exc.status_code}): {exc.message}")
            st.stop()
        except Exception as exc:
            show_error(exc)
            st.stop()

    st.success(f"Provider: {result.get('provider')} - Model: {result.get('model')}")
    st.subheader(t("executive_summary"))
    st.info(result.get("executive_summary") or t("no_summary"))

    def _render_list(title: str, items: list[str], icon: str) -> None:
        if not items:
            return
        st.subheader(f"{icon} {title}")
        for item in items:
            st.markdown(f"- {item}")

    _render_list(t("insights"), result.get("insights") or [], "💡")
    _render_list(t("anomalies"), result.get("anomalies") or [], "⚠️")
    _render_list(t("suggestions"), result.get("suggestions") or [], "✅")
    _render_list(t("risks"), result.get("risks") or [], "🚨")

    raw = result.get("raw_llm_response")
    if raw:
        with st.expander(t("raw_response")):
            st.text(raw)
