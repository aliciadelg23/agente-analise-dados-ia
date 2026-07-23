"""Landing page for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIError
from dashboard.theme import apply_page_config, get_api_client

apply_page_config("Dashboard - Agente de Analise de Dados com IA")

st.title("Agente de Analise de Dados com IA")
st.caption("Dashboard operacional sobre a API de analise de dados.")

client = get_api_client()

col_status, col_dataset = st.columns([2, 1])

with col_status:
    st.subheader("Status da API")
    try:
        info = client.info()
        health = client.health()
        st.success(f"API online em `{client.base_url}`")
        st.markdown(
            f"- **Nome**: {info.get('name', 'n/a')}\n"
            f"- **Versao**: {info.get('version', 'n/a')}\n"
            f"- **Ambiente**: {info.get('environment', 'n/a')}\n"
            f"- **Health**: {health.get('status', 'unknown')}"
        )
    except APIError as exc:
        st.error(f"API respondeu com erro ({exc.status_code}): {exc.message}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Falha ao contactar a API: {exc}")

with col_dataset:
    st.subheader("Dataset ativo")
    dataset_id = st.session_state.get("dataset_id")
    if dataset_id:
        st.info(dataset_id)
        st.caption(
            f"Rows: {st.session_state.get('dataset_rows', '?')} - "
            f"Cols: {st.session_state.get('dataset_columns', '?')}"
        )
    else:
        st.warning("Nenhum dataset carregado. Va para **Upload**.")

st.divider()

st.subheader("Como usar")
st.markdown(
    """
    1. Envie um CSV em **Upload**.
    2. Explore metadados e graficos em **Visualizations**.
    3. Treine modelos em **Training** e inspecione em **Explanations**.
    4. Peca insights em linguagem natural em **Insights**.
    5. Converse com o agente na aba **Chat**.
    """
)
