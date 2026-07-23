"""Landing page for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIError
from dashboard.i18n import t
from dashboard.theme import apply_page_config, get_api_client, render_settings_sidebar

apply_page_config()
render_settings_sidebar()

st.title(t("app_title"))
st.caption(t("app_caption"))

client = get_api_client()

col_status, col_dataset = st.columns([2, 1])

with col_status:
    st.subheader(t("api_status"))
    try:
        info = client.info()
        health = client.health()
        st.success(f"{t('api_online')} `{client.base_url}`")
        st.markdown(
            f"- **{t('name')}**: {info.get('name', 'n/a')}\n"
            f"- **{t('version')}**: {info.get('version', 'n/a')}\n"
            f"- **{t('environment')}**: {info.get('environment', 'n/a')}\n"
            f"- **{t('health')}**: {health.get('status', 'unknown')}"
        )
    except APIError as exc:
        st.error(f"{t('api_error')} ({exc.status_code}): {exc.message}")
    except Exception as exc:
        st.error(f"{t('api_unreachable')}: {exc}")

with col_dataset:
    st.subheader(t("active_dataset"))
    dataset_id = st.session_state.get("dataset_id")
    if dataset_id:
        st.info(dataset_id)
        st.caption(
            f"{t('rows')}: {st.session_state.get('dataset_rows', '?')} - "
            f"{t('columns')}: {st.session_state.get('dataset_columns', '?')}"
        )
    else:
        st.warning(t("no_dataset"))

st.divider()

st.subheader(t("how_to_use"))
st.markdown(t("how_to_use_body"))
