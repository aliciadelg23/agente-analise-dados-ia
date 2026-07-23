"""Visualizations page: EDA summary + generated charts."""

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

st.title(t("viz_title"))
st.caption(t("viz_caption"))

dataset_id = require_dataset_id()
if dataset_id is None:
    st.stop()

client = get_api_client()

with st.spinner(t("loading_summary")):
    try:
        summary = client.summary(dataset_id)
    except APIError as exc:
        st.error(f"{t('api_error')} ({exc.status_code}): {exc.message}")
        st.stop()
    except Exception as exc:
        show_error(exc)
        st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric(t("rows"), summary.get("rows", "?"))
col2.metric(t("columns"), summary.get("columns", "?"))
col3.metric(t("memory"), summary.get("memory", "?"))
col4.metric(t("duplicates"), summary.get("duplicates", "?"))

with st.expander(t("columns_and_types"), expanded=False):
    dtypes = summary.get("dtypes", {}) or {}
    if dtypes:
        st.dataframe(
            {"column": list(dtypes.keys()), "dtype": list(dtypes.values())},
            hide_index=True,
            use_container_width=True,
        )

with st.expander(t("null_values"), expanded=False):
    null_counts = summary.get("null_counts", {}) or {}
    null_pct = summary.get("null_percentages", {}) or {}
    if null_counts:
        st.dataframe(
            {
                "column": list(null_counts.keys()),
                "null_count": list(null_counts.values()),
                "null_pct": [null_pct.get(name, 0) for name in null_counts],
            },
            hide_index=True,
            use_container_width=True,
        )

st.divider()

st.subheader(t("charts"))
with st.spinner(t("generating_charts")):
    try:
        charts = client.charts(dataset_id)
    except APIError as exc:
        st.error(f"{t('api_error')} ({exc.status_code}): {exc.message}")
        st.stop()
    except Exception as exc:
        show_error(exc)
        st.stop()

charts_payload = charts.get("charts", {}) or {}
image_base = client.base_url


def _abs_url(path: str) -> str:
    if not path:
        return ""
    return path if path.startswith("http") else f"{image_base}{path}"


def _render_column_charts(title: str, items: list[dict]) -> None:
    if not items:
        return
    st.markdown(f"**{title}**")
    cols = st.columns(min(3, len(items)))
    for i, item in enumerate(items):
        with cols[i % len(cols)]:
            st.caption(item.get("column", ""))
            st.image(_abs_url(item.get("png_url", "")), use_container_width=True)


_render_column_charts(t("histograms"), charts_payload.get("histograms") or [])
_render_column_charts(t("boxplots"), charts_payload.get("boxplots") or [])
_render_column_charts(t("bar_charts"), charts_payload.get("bar_charts") or [])

heatmap = charts_payload.get("correlation_heatmap")
if heatmap:
    st.markdown(f"**{t('correlation_heatmap')}**")
    st.image(_abs_url(heatmap.get("png_url", "")), use_container_width=True)

distribution = charts_payload.get("category_distributions")
if distribution:
    st.markdown(f"**{t('category_distribution')}**")
    st.image(_abs_url(distribution.get("png_url", "")), use_container_width=True)
