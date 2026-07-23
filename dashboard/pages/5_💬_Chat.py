"""Chat page: talk to the LangChain agent that uses the app's tools."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIError
from dashboard.i18n import t
from dashboard.theme import apply_page_config, get_api_client, render_settings_sidebar, show_error

apply_page_config()
render_settings_sidebar()

st.title(t("chat_title"))
st.caption(t("chat_caption"))

client = get_api_client()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for entry in st.session_state["chat_history"]:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

dataset_id = st.session_state.get("dataset_id")
if dataset_id:
    st.caption(t("chat_active_dataset", did=dataset_id))
else:
    st.caption(t("chat_no_dataset"))

user_input = st.chat_input(t("chat_placeholder"))
if user_input:
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner(t("chat_thinking")):
        try:
            response = client.chat({"query": user_input})
            output = response.get("output") or t("chat_no_answer")
        except APIError as exc:
            output = f"❌ {t('api_error')} ({exc.status_code}): {exc.message}"
        except Exception as exc:
            show_error(exc)
            output = f"❌ {exc}"

    st.session_state["chat_history"].append({"role": "assistant", "content": output})
    with st.chat_message("assistant"):
        st.markdown(output)

if st.button(t("clear_history")):
    st.session_state["chat_history"] = []
    st.rerun()
