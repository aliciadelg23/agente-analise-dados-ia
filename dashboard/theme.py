"""Shared UI helpers used across every dashboard page."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIClient


PAGE_CONFIG = {
    "page_title": "Agente de Analise de Dados com IA",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}


def apply_page_config(page_title: str | None = None) -> None:
    """Apply the shared Streamlit page config.

    ``st.set_page_config`` must be the first Streamlit call in every
    page module. Passing ``page_title`` overrides only the tab title.
    """
    config = dict(PAGE_CONFIG)
    if page_title:
        config["page_title"] = page_title
    st.set_page_config(**config)


def get_api_client() -> APIClient:
    """Return a cached APIClient stored in session state."""
    if "api_client" not in st.session_state:
        st.session_state["api_client"] = APIClient()
    return st.session_state["api_client"]


def require_dataset_id() -> str | None:
    """Read the current dataset_id from session state, warn if missing."""
    dataset_id = st.session_state.get("dataset_id")
    if not dataset_id:
        st.info(
            "No dataset selected yet. Upload a CSV on the **Upload** page first, "
            "then come back here."
        )
        return None
    return dataset_id


def show_error(exc: Exception) -> None:
    """Render an API error consistently."""
    st.error(f"Request failed: {exc}")
