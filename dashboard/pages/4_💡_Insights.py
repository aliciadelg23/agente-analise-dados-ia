"""Insights page: LLM-authored structured analysis."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIError
from dashboard.theme import apply_page_config, get_api_client, require_dataset_id, show_error

apply_page_config("Insights - Dashboard")

st.title("Insights gerados por IA")
st.caption("EDA + LLM produzem resumo executivo, insights, anomalias, sugestoes e riscos.")

dataset_id = require_dataset_id()
if dataset_id is None:
    st.stop()

client = get_api_client()

with st.form("insights-form"):
    col_prov, col_model = st.columns(2)
    provider = col_prov.selectbox(
        "Provider (opcional)",
        options=["", "openai", "anthropic", "gemini"],
        format_func=lambda v: v or "default",
    )
    model = col_model.text_input("Modelo (opcional)")
    submitted = st.form_submit_button("Gerar insights")

if submitted:
    body: dict = {}
    if provider:
        body["provider"] = provider
    if model:
        body["model"] = model
    with st.spinner("Consultando LLM..."):
        try:
            result = client.insights(dataset_id, body)
        except APIError as exc:
            st.error(f"API retornou {exc.status_code}: {exc.message}")
            st.stop()
        except Exception as exc:
            show_error(exc)
            st.stop()

    st.success(f"Provider: {result.get('provider')} - Modelo: {result.get('model')}")
    st.subheader("Resumo executivo")
    st.info(result.get("executive_summary") or "Sem resumo.")

    def _render_list(title: str, items: list[str], icon: str) -> None:
        if not items:
            return
        st.subheader(f"{icon} {title}")
        for item in items:
            st.markdown(f"- {item}")

    _render_list("Insights", result.get("insights") or [], "💡")
    _render_list("Anomalias", result.get("anomalies") or [], "⚠️")
    _render_list("Sugestoes", result.get("suggestions") or [], "✅")
    _render_list("Riscos", result.get("risks") or [], "🚨")

    raw = result.get("raw_llm_response")
    if raw:
        with st.expander("Resposta bruta (parse falhou)"):
            st.text(raw)
