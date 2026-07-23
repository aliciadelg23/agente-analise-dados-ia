"""Upload page: send a CSV file to the API and store the dataset id."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIError
from dashboard.i18n import t
from dashboard.theme import apply_page_config, get_api_client, render_settings_sidebar, show_error

apply_page_config()
render_settings_sidebar()

st.title(t("upload_title"))
st.caption(t("upload_caption"))

client = get_api_client()

uploaded = st.file_uploader(t("upload_select"), type=["csv"])

if uploaded is not None:
    with st.spinner(t("upload_sending")):
        try:
            response = client.upload_dataset(uploaded.name, uploaded.getvalue())
        except APIError as exc:
            st.error(f"{t('api_error')} ({exc.status_code}): {exc.message}")
            response = None
        except Exception as exc:
            show_error(exc)
            response = None

    if response is not None:
        st.session_state["dataset_id"] = response.get("dataset_id")
        st.session_state["dataset_rows"] = response.get("rows")
        st.session_state["dataset_columns"] = response.get("columns")
        st.success(t("upload_done"))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(t("rows"), response.get("rows", "?"))
        col2.metric(t("columns"), response.get("columns", "?"))
        col3.metric(t("size"), response.get("size", "?"))
        col4.metric(t("encoding"), response.get("encoding", "?"))
        st.caption(
            f"{t('dataset_id')}: `{response.get('dataset_id', '?')}` - "
            f"{t('separator')}: `{response.get('separator', '?')}`"
        )

active = st.session_state.get("dataset_id")
if active and uploaded is None:
    st.info(f"{t('active_dataset')}: `{active}`")
