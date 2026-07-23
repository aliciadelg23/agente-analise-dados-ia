"""Upload page: send a CSV file to the API and store the dataset id."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIError
from dashboard.theme import apply_page_config, get_api_client, show_error

apply_page_config("Upload - Dashboard")

st.title("Upload de dataset")
st.caption("Envie um arquivo CSV para o backend.")

client = get_api_client()

uploaded = st.file_uploader("Selecione um CSV", type=["csv"])

if uploaded is not None:
    with st.spinner("Enviando..."):
        try:
            response = client.upload_dataset(uploaded.name, uploaded.getvalue())
        except APIError as exc:
            st.error(f"API retornou {exc.status_code}: {exc.message}")
            response = None
        except Exception as exc:
            show_error(exc)
            response = None

    if response is not None:
        st.session_state["dataset_id"] = response.get("dataset_id")
        st.session_state["dataset_rows"] = response.get("rows")
        st.session_state["dataset_columns"] = response.get("columns")
        st.success("Upload concluido.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", response.get("rows", "?"))
        col2.metric("Columns", response.get("columns", "?"))
        col3.metric("Size", response.get("size", "?"))
        col4.metric("Encoding", response.get("encoding", "?"))
        st.caption(
            f"Dataset ID: `{response.get('dataset_id', '?')}` - "
            f"Separator: `{response.get('separator', '?')}`"
        )

active = st.session_state.get("dataset_id")
if active and uploaded is None:
    st.info(f"Dataset ativo: `{active}`. Faca upload de outro CSV para trocar.")
