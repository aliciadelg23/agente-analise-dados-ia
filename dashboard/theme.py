"""Shared UI helpers used across every dashboard page."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIClient
from dashboard.i18n import DEFAULT_LANGUAGE, LANGUAGES, t
from dashboard.theming import DEFAULT_THEME, THEMES, apply_theme

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


def render_settings_sidebar() -> None:
    """Render language and theme controls in the sidebar.

    Must be called on every page (after apply_page_config) so the
    preferences are always visible and applied.
    """
    st.session_state.setdefault("language", DEFAULT_LANGUAGE)
    st.session_state.setdefault("theme", DEFAULT_THEME)

    with st.sidebar:
        st.markdown(f"### {t('sidebar_settings')}")
        language_codes = list(LANGUAGES.keys())
        current_language = st.session_state["language"]
        try:
            language_index = language_codes.index(current_language)
        except ValueError:
            language_index = 0
        selected_language = st.selectbox(
            t("sidebar_language"),
            options=language_codes,
            index=language_index,
            format_func=lambda code: LANGUAGES[code],
            key="language_selector",
        )
        if selected_language != st.session_state["language"]:
            st.session_state["language"] = selected_language
            st.rerun()

        theme_codes = list(THEMES.keys())
        theme_labels = {"dark": t("theme_dark"), "light": t("theme_light")}
        current_theme = st.session_state["theme"]
        try:
            theme_index = theme_codes.index(current_theme)
        except ValueError:
            theme_index = 0
        selected_theme = st.selectbox(
            t("sidebar_theme"),
            options=theme_codes,
            index=theme_index,
            format_func=lambda code: theme_labels.get(code, code),
            key="theme_selector",
        )
        if selected_theme != st.session_state["theme"]:
            st.session_state["theme"] = selected_theme
            st.rerun()

    apply_theme()


def require_dataset_id() -> str | None:
    """Read the current dataset_id from session state, warn if missing."""
    dataset_id = st.session_state.get("dataset_id")
    if not dataset_id:
        st.info(t("require_dataset"))
        return None
    return dataset_id


def show_error(exc: Exception) -> None:
    """Render an API error consistently."""
    st.error(f"Request failed: {exc}")
