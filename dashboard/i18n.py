"""Small in-memory i18n layer for the dashboard.

Every user-facing string used by the pages is registered in
``_TRANSLATIONS`` under a stable key. Pages call ``t(key)`` and get
the string in the currently selected language, defaulting to
Portuguese when a key is missing.
"""

from __future__ import annotations

import streamlit as st

LANGUAGES = {
    "pt": "Português",
    "en": "English",
}
DEFAULT_LANGUAGE = "pt"

_TRANSLATIONS: dict[str, dict[str, str]] = {
    "app_title": {
        "pt": "Agente de Análise de Dados com IA",
        "en": "AI Data Analysis Agent",
    },
    "app_caption": {
        "pt": "Dashboard operacional sobre a API de análise de dados.",
        "en": "Operational dashboard for the data analysis API.",
    },
    "sidebar_settings": {"pt": "Preferências", "en": "Settings"},
    "sidebar_language": {"pt": "Idioma", "en": "Language"},
    "sidebar_theme": {"pt": "Tema", "en": "Theme"},
    "theme_dark": {"pt": "Escuro", "en": "Dark"},
    "theme_light": {"pt": "Claro", "en": "Light"},
    "api_status": {"pt": "Status da API", "en": "API status"},
    "api_online": {"pt": "API online em", "en": "API online at"},
    "api_error": {"pt": "API respondeu com erro", "en": "API returned an error"},
    "api_unreachable": {"pt": "Falha ao contactar a API", "en": "Could not reach the API"},
    "name": {"pt": "Nome", "en": "Name"},
    "version": {"pt": "Versão", "en": "Version"},
    "environment": {"pt": "Ambiente", "en": "Environment"},
    "health": {"pt": "Health", "en": "Health"},
    "active_dataset": {"pt": "Dataset ativo", "en": "Active dataset"},
    "no_dataset": {
        "pt": "Nenhum dataset carregado. Vá para **Upload**.",
        "en": "No dataset loaded yet. Go to **Upload**.",
    },
    "require_dataset": {
        "pt": "Nenhum dataset selecionado. Envie um CSV na página **Upload** primeiro.",
        "en": "No dataset selected yet. Upload a CSV on the **Upload** page first.",
    },
    "how_to_use": {"pt": "Como usar", "en": "How to use"},
    "how_to_use_body": {
        "pt": (
            "1. Envie um CSV em **Upload**.\n"
            "2. Explore metadados e gráficos em **Visualizations**.\n"
            "3. Treine modelos em **Training** e inspecione em **Explanations**.\n"
            "4. Peça insights em linguagem natural em **Insights**.\n"
            "5. Converse com o agente na aba **Chat**."
        ),
        "en": (
            "1. Upload a CSV on **Upload**.\n"
            "2. Explore metadata and charts on **Visualizations**.\n"
            "3. Train models on **Training** and inspect them on **Explanations**.\n"
            "4. Ask for natural-language insights on **Insights**.\n"
            "5. Talk to the agent on **Chat**."
        ),
    },
    "upload_title": {"pt": "Upload de dataset", "en": "Upload dataset"},
    "upload_caption": {
        "pt": "Envie um arquivo CSV para o backend.",
        "en": "Send a CSV file to the backend.",
    },
    "upload_select": {"pt": "Selecione um CSV", "en": "Select a CSV"},
    "upload_sending": {"pt": "Enviando...", "en": "Uploading..."},
    "upload_done": {"pt": "Upload concluído.", "en": "Upload complete."},
    "rows": {"pt": "Linhas", "en": "Rows"},
    "columns": {"pt": "Colunas", "en": "Columns"},
    "size": {"pt": "Tamanho", "en": "Size"},
    "encoding": {"pt": "Encoding", "en": "Encoding"},
    "dataset_id": {"pt": "Dataset ID", "en": "Dataset ID"},
    "separator": {"pt": "Separador", "en": "Separator"},
    "viz_title": {"pt": "Visualizações", "en": "Visualizations"},
    "viz_caption": {
        "pt": "Estatísticas descritivas e gráficos gerados pela API.",
        "en": "Descriptive statistics and charts produced by the API.",
    },
    "loading_summary": {"pt": "Carregando resumo...", "en": "Loading summary..."},
    "memory": {"pt": "Memória", "en": "Memory"},
    "duplicates": {"pt": "Duplicados", "en": "Duplicates"},
    "columns_and_types": {"pt": "Colunas e tipos", "en": "Columns and types"},
    "null_values": {"pt": "Valores nulos por coluna", "en": "Null values per column"},
    "charts": {"pt": "Gráficos", "en": "Charts"},
    "generating_charts": {"pt": "Gerando gráficos...", "en": "Generating charts..."},
    "histograms": {"pt": "Histogramas", "en": "Histograms"},
    "boxplots": {"pt": "Boxplots", "en": "Boxplots"},
    "bar_charts": {"pt": "Gráficos de barras", "en": "Bar charts"},
    "correlation_heatmap": {"pt": "Heatmap de correlação", "en": "Correlation heatmap"},
    "category_distribution": {
        "pt": "Distribuição das categorias",
        "en": "Category distribution",
    },
    "training_title": {"pt": "Treinamento de modelos", "en": "Model training"},
    "training_caption": {
        "pt": "Dispara o pipeline de ML e mostra as métricas dos candidatos.",
        "en": "Runs the ML pipeline and reports metrics for every candidate.",
    },
    "loading_columns": {"pt": "Carregando colunas...", "en": "Loading columns..."},
    "target_column": {"pt": "Coluna alvo (target)", "en": "Target column"},
    "problem_type": {"pt": "Tipo de problema", "en": "Problem type"},
    "test_size": {"pt": "Test size", "en": "Test size"},
    "cv_folds": {"pt": "CV folds", "en": "CV folds"},
    "train_button": {"pt": "Treinar", "en": "Train"},
    "training_running": {
        "pt": "Treinando... isso pode levar alguns segundos",
        "en": "Training... this may take a few seconds",
    },
    "winner": {"pt": "Vencedor", "en": "Winner"},
    "samples_train": {"pt": "Amostras (train)", "en": "Samples (train)"},
    "samples_test": {"pt": "Amostras (test)", "en": "Samples (test)"},
    "model_id": {"pt": "Model ID", "en": "Model ID"},
    "candidate_metrics": {
        "pt": "Métricas dos candidatos",
        "en": "Candidate metrics",
    },
    "model_saved_at": {"pt": "Modelo salvo em", "en": "Model saved at"},
    "insights_title": {"pt": "Insights gerados por IA", "en": "AI-generated insights"},
    "insights_caption": {
        "pt": "EDA + LLM produzem resumo executivo, insights, anomalias, sugestões e riscos.",
        "en": "EDA + LLM produce an executive summary, insights, anomalies, suggestions, and risks.",
    },
    "provider_optional": {
        "pt": "Provider (opcional)",
        "en": "Provider (optional)",
    },
    "model_optional": {"pt": "Modelo (opcional)", "en": "Model (optional)"},
    "generate_insights": {"pt": "Gerar insights", "en": "Generate insights"},
    "asking_llm": {"pt": "Consultando LLM...", "en": "Asking the LLM..."},
    "executive_summary": {"pt": "Resumo executivo", "en": "Executive summary"},
    "no_summary": {"pt": "Sem resumo.", "en": "No summary."},
    "insights": {"pt": "Insights", "en": "Insights"},
    "anomalies": {"pt": "Anomalias", "en": "Anomalies"},
    "suggestions": {"pt": "Sugestões", "en": "Suggestions"},
    "risks": {"pt": "Riscos", "en": "Risks"},
    "raw_response": {
        "pt": "Resposta bruta (parse falhou)",
        "en": "Raw response (parse failed)",
    },
    "chat_title": {"pt": "Conversa com o agente", "en": "Chat with the agent"},
    "chat_caption": {
        "pt": "O agente LangChain tem acesso a 5 ferramentas: dataset info, EDA, statistics, ML, charts.",
        "en": "The LangChain agent has access to 5 tools: dataset info, EDA, statistics, ML, charts.",
    },
    "chat_active_dataset": {
        "pt": "Dataset ativo: `{did}` (mencione o id nas perguntas quando relevante)",
        "en": "Active dataset: `{did}` (mention it in your prompt when relevant)",
    },
    "chat_no_dataset": {
        "pt": "Nenhum dataset carregado; o agente ainda funciona mas não tem contexto local.",
        "en": "No dataset loaded; the agent still works but has no local context.",
    },
    "chat_placeholder": {
        "pt": "Pergunte algo ao agente...",
        "en": "Ask the agent something...",
    },
    "chat_thinking": {"pt": "Agente pensando...", "en": "Agent thinking..."},
    "chat_no_answer": {"pt": "(sem resposta)", "en": "(no answer)"},
    "clear_history": {"pt": "Limpar histórico", "en": "Clear history"},
    "explain_title": {"pt": "Explicabilidade", "en": "Explainability"},
    "explain_caption": {
        "pt": "Feature importance + SHAP values do modelo treinado.",
        "en": "Feature importance + SHAP values for the trained model.",
    },
    "explain_no_model": {
        "pt": "Treine um modelo em **Training** ou informe um Model ID manualmente.",
        "en": "Train a model on **Training** or enter a Model ID manually.",
    },
    "explain_button": {"pt": "Explicar modelo", "en": "Explain model"},
    "computing_shap": {"pt": "Calculando SHAP...", "en": "Computing SHAP..."},
    "algorithm": {"pt": "Algoritmo", "en": "Algorithm"},
    "target": {"pt": "Target", "en": "Target"},
    "feature_importance": {"pt": "Feature importance", "en": "Feature importance"},
    "shap_mean_abs": {
        "pt": "SHAP - valores médios absolutos",
        "en": "SHAP - mean absolute values",
    },
    "shap_summary_plot": {"pt": "SHAP summary plot", "en": "SHAP summary plot"},
}


def get_language() -> str:
    """Return the currently selected language code, defaulting to Portuguese."""
    return st.session_state.get("language", DEFAULT_LANGUAGE)


def t(key: str, **fmt: object) -> str:
    """Look up ``key`` in the selected language.

    Missing keys fall back to Portuguese; unknown keys return the key
    itself so translations are easy to spot in the UI.
    """
    language = get_language()
    entry = _TRANSLATIONS.get(key)
    if entry is None:
        return key
    text = entry.get(language) or entry.get(DEFAULT_LANGUAGE) or key
    if fmt:
        try:
            return text.format(**fmt)
        except (KeyError, IndexError):
            return text
    return text
