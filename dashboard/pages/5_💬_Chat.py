"""Chat page: talk to the LangChain agent that uses the app's tools."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIError
from dashboard.theme import apply_page_config, get_api_client, show_error

apply_page_config("Chat - Dashboard")

st.title("Conversa com o agente")
st.caption(
    "O agente LangChain tem acesso a 5 ferramentas: dataset info, EDA, statistics, ML, charts."
)

client = get_api_client()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for entry in st.session_state["chat_history"]:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

dataset_id = st.session_state.get("dataset_id")
if dataset_id:
    st.caption(f"Dataset ativo: `{dataset_id}` (mencione o id nas perguntas quando relevante)")
else:
    st.caption("Nenhum dataset carregado; o agente ainda funciona mas nao tem contexto local.")

user_input = st.chat_input("Pergunte algo ao agente...")
if user_input:
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Agente pensando..."):
        try:
            response = client.chat({"query": user_input})
            output = response.get("output") or "(sem resposta)"
        except APIError as exc:
            output = f"❌ API retornou {exc.status_code}: {exc.message}"
        except Exception as exc:
            show_error(exc)
            output = f"❌ Falha: {exc}"

    st.session_state["chat_history"].append({"role": "assistant", "content": output})
    with st.chat_message("assistant"):
        st.markdown(output)

if st.button("Limpar historico"):
    st.session_state["chat_history"] = []
    st.rerun()
