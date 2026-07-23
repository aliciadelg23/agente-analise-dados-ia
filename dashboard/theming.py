"""Lightweight runtime theme switcher.

Streamlit's built-in theming is only picked up at page-config time,
so this module injects a small CSS block on every page to override
the colors after the initial render. The overrides target the
containers Streamlit exposes with stable classnames.
"""

from __future__ import annotations

import streamlit as st

THEMES = {
    "dark": {
        "background": "#0e1117",
        "surface": "#1a1f2b",
        "text": "#e5e7eb",
        "muted": "#94a3b8",
        "accent": "#4c8bf5",
    },
    "light": {
        "background": "#f8fafc",
        "surface": "#ffffff",
        "text": "#0f172a",
        "muted": "#475569",
        "accent": "#2563eb",
    },
}
DEFAULT_THEME = "dark"


def get_theme() -> str:
    """Return the active theme name, defaulting to dark."""
    return st.session_state.get("theme", DEFAULT_THEME)


def _build_css(theme: str) -> str:
    palette = THEMES.get(theme, THEMES[DEFAULT_THEME])
    return f"""
    <style>
    :root {{
        --app-bg: {palette["background"]};
        --app-surface: {palette["surface"]};
        --app-text: {palette["text"]};
        --app-muted: {palette["muted"]};
        --app-accent: {palette["accent"]};
    }}
    section[data-testid="stAppViewContainer"] > .main,
    section[data-testid="stAppViewContainer"] > .main .block-container,
    [data-testid="stAppViewContainer"] {{
        background-color: var(--app-bg) !important;
        color: var(--app-text) !important;
    }}
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {{
        background-color: var(--app-surface) !important;
        color: var(--app-text) !important;
    }}
    h1, h2, h3, h4, h5, h6, p, label, span, div,
    [data-testid="stMarkdownContainer"] {{
        color: var(--app-text) !important;
    }}
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] span,
    small {{
        color: var(--app-muted) !important;
    }}
    [data-testid="stMetric"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {{
        color: var(--app-text) !important;
    }}
    button[kind], .stButton > button, .stDownloadButton > button,
    [data-testid="baseButton-secondary"] {{
        background-color: var(--app-accent) !important;
        color: #ffffff !important;
        border: 1px solid var(--app-accent) !important;
    }}
    a {{ color: var(--app-accent) !important; }}
    </style>
    """


def apply_theme() -> None:
    """Inject the CSS for the currently selected theme."""
    st.markdown(_build_css(get_theme()), unsafe_allow_html=True)
