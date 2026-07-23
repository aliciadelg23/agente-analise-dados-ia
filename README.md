# Agente de Análise de Dados com IA

Plataforma de análise de dados orquestrada por agentes de IA. Este repositório contém apenas o esqueleto arquitetural: as funcionalidades de negócio serão adicionadas em etapas seguintes.

## Stack

- Python 3.12+
- FastAPI + Uvicorn
- Pydantic + pydantic-settings
- uv (gerenciamento de dependências e ambiente virtual)
- Ruff + Black (lint e formatação)
- Pytest (testes)
- GitHub Actions (CI)

## Estrutura de pastas

```text
app/
    api/            # Entradas HTTP (routers, dependências, schemas)
    core/           # Preocupações transversais (logging, erros, segurança)
    config/         # Configuração e carga de variáveis de ambiente
    models/         # Modelos de domínio e schemas Pydantic
    services/       # Regras de negócio e orquestração
    repositories/   # Camada de acesso a dados
    agents/         # Agentes de IA especializados
    pipelines/      # Pipelines de processamento de dados
    llms/           # Adaptadores para provedores de LLM
    utils/          # Utilitários compartilhados

tests/              # Testes automatizados
docs/               # Documentação adicional
scripts/            # Scripts utilitários
.github/workflows/  # Pipelines de CI
```

A separação por responsabilidade permite crescer sem misturar camadas: a API não conhece o repositório, o serviço não conhece o transporte HTTP, e os agentes dependem de abstrações de LLM em vez de SDKs concretos.

## Requisitos

- Python 3.12 ou superior
- [uv](https://docs.astral.sh/uv/) instalado

Instalação do uv (Linux/macOS):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup local

```bash
# Clonar o repositório
git clone https://github.com/aliciadelg23/agente-analise-dados-ia.git
cd agente-analise-dados-ia

# Copiar variáveis de ambiente
cp .env.example .env

# Instalar dependências (cria .venv automaticamente)
uv sync --all-groups
```

## Executar a aplicação

```bash
uv run uvicorn app.main:app --reload
```

A API sobe em `http://localhost:8000`.

### Endpoints disponíveis

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/` | Informações da API (nome, versão, ambiente). |
| `GET` | `/health` | Status de liveness. |
| `POST` | `/datasets/upload` | Recebe um CSV, valida, persiste e retorna metadados. |
| `GET` | `/datasets/{dataset_id}/summary` | Análise exploratória (EDA) do dataset armazenado. |
| `POST` | `/datasets/{dataset_id}/clean` | Limpa o dataset e persiste uma nova versão. |
| `GET` | `/datasets/{dataset_id}/charts` | Gera gráficos (histograma, boxplot, heatmap, barras, distribuição). |
| `POST` | `/datasets/{dataset_id}/train` | Treina modelos de ML e retorna métricas + modelo escolhido. |
| `GET` | `/models/{model_id}/explain` | Explicabilidade: feature importance, SHAP e gráfico. |
| `POST` | `/datasets/{dataset_id}/insights` | Insights gerados por LLM sobre o dataset (resumo, anomalias, sugestões, riscos). |
| `POST` | `/agent/chat` | Agente LangChain com acesso a 5 ferramentas (dataset, EDA, estatísticas, ML, gráficos). |

### Exemplos

```bash
# Info
curl http://localhost:8000/
# {"name":"Agente de Analise de Dados com IA","version":"0.1.0",...}

# Health
curl http://localhost:8000/health
# {"status":"ok","version":"0.1.0"}

# Upload de CSV
curl -X POST http://localhost:8000/datasets/upload \
  -F "file=@caminho/para/dataset.csv"
# {
#   "dataset_id": "...",
#   "filename": "dataset.csv",
#   "rows": 1250,
#   "columns": 18,
#   "size": "1.2 MB",
#   "uploaded_at": "2026-07-23T00:00:00Z",
#   "encoding": "utf-8",
#   "separator": ","
# }

# Análise exploratória (EDA) do dataset
curl http://localhost:8000/datasets/<dataset_id>/summary
# {
#   "dataset_id": "...",
#   "rows": 1250,
#   "columns": 18,
#   "memory": "412.5 KB",
#   "duplicates": 3,
#   "null_counts": {"age": 12, "city": 0, ...},
#   "null_percentages": {"age": 0.96, "city": 0.0, ...},
#   "dtypes": {"age": "float64", "city": "object", ...},
#   "numeric_columns": ["age", "salary"],
#   "categorical_columns": ["city", "role"],
#   "numeric_stats": {
#     "age": {
#       "mean": 32.4, "median": 30.0, "min": 18, "max": 65,
#       "std": 8.7, "q25": 25.0, "q50": 30.0, "q75": 40.0
#     }
#   },
#   "categorical_stats": {
#     "city": {
#       "unique_count": 42,
#       "top_values": [{"value": "Lisbon", "count": 128}, ...]
#     }
#   }
# }
```

Documentação interativa gerada pelo FastAPI:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

### Upload de datasets

- Extensão aceita: `.csv`.
- Tamanho máximo: definido por `MAX_UPLOAD_SIZE_MB` (padrão 50 MB).
- Arquivos são salvos em `storage/uploads/{dataset_id}.csv` (diretório configurável via `STORAGE_DIR`).
- O CSV é inspecionado com Pandas após o upload: detecção automática de encoding (utf-8, latin-1, ...), separador (`,` `;` `|` tab) e tipos de coluna.
- Respostas de erro seguem o formato `{"error": {"code": "...", "message": "..."}}`.

### Agente LangChain

O endpoint `POST /agent/chat` expõe um agente LangChain (`create_agent`, LangChain 1.x) que decide autonomamente quais ferramentas invocar para responder uma pergunta em linguagem natural.

**Ferramentas disponíveis** (em `app/agents/tools.py`):

| Nome | O que faz |
|------|-----------|
| `dataset_info` | Confirma que um `dataset_id` existe e retorna metadados básicos (path, size). |
| `dataset_summary` | Roda o EDA completo (mesmo do endpoint da Etapa 3). |
| `column_statistics` | Retorna estatísticas focadas em uma coluna específica (numérica ou categórica). |
| `train_model` | Dispara o pipeline de ML da Etapa 6 e retorna as métricas dos candidatos. |
| `generate_charts` | Gera o conjunto de gráficos da Etapa 5 e retorna as URLs. |

Cada ferramenta valida seus inputs via Pydantic e retorna uma string JSON — formato que os agentes LangChain esperam. Erros das ferramentas viram JSON também (`{"error": {"code": ..., "message": ...}}`), então uma falha nunca quebra o loop do agente.

**LLM utilizado**: `ChatOpenAI` (`langchain-openai`). Configurável via `OPENAI_API_KEY` e `OPENAI_MODEL` do `.env`. O prompt system pede ao modelo que use as ferramentas antes de responder.

**Request**:

```json
{
  "query": "Qual coluna tem mais valores nulos no dataset 5c1a...?",
  "model": "gpt-4o-mini"
}
```

- `query` — obrigatório, pergunta em linguagem natural.
- `model` — opcional, sobrescreve `OPENAI_MODEL`.

**Response**:

```json
{
  "output": "A coluna 'email' tem 12% de valores nulos, seguida por 'salary' com 3%.",
  "model": "gpt-4o-mini"
}
```

**Erros**:

- `400 missing_llm_credentials` — `OPENAI_API_KEY` ausente.

Exemplo:

```bash
curl -X POST http://localhost:8000/agent/chat \
  -H "content-type: application/json" \
  -d '{"query": "Faça um resumo do dataset 5c1a-... e liste 3 riscos de qualidade de dados."}'
```

**Notas**:

- O agente pode encadear múltiplas ferramentas para responder — por exemplo, chamar `dataset_info` para confirmar o id, `dataset_summary` para obter EDA, e voltar ao LLM para escrever a resposta.
- Nenhuma chamada real ao OpenAI é feita nos testes automatizados (todos os testes mockam `ChatOpenAI` e `create_agent`).

### Insights gerados por LLM

O endpoint `POST /datasets/{dataset_id}/insights` combina a análise exploratória da Etapa 3 com a camada de LLMs da Etapa 8 para produzir uma análise em linguagem natural.

**Fluxo interno**:

1. `EDAService.summarize(dataset_id)` gera as estatísticas descritivas.
2. A saída é serializada em JSON compacto.
3. Uma mensagem system dita o formato de resposta obrigatório; uma mensagem user carrega o JSON da EDA.
4. `LLMProvider.chat(...)` é chamado via o factory da camada de LLMs.
5. O texto retornado é interpretado como JSON; se necessário, um regex extrai o bloco JSON de uma resposta "conversacional".

**Request** (todos os campos opcionais):

```json
{
  "provider": "anthropic",
  "model": "claude-opus-4-8"
}
```

- Body vazio (`{}`) → usa `DEFAULT_LLM_PROVIDER` e o modelo padrão do provider.
- `provider` — sobrescreve o provider padrão. Valores aceitos: `openai`, `anthropic`, `gemini`.
- `model` — sobrescreve o modelo do provider escolhido.

**Response**:

```json
{
  "dataset_id": "...",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "executive_summary": "Dataset has 1250 rows and 18 columns...",
  "insights": [
    "Age distribution is right-skewed with mean ~35 years.",
    "Salary correlates positively with age (r=0.72)."
  ],
  "anomalies": ["Column 'email' has 12% nulls..."],
  "suggestions": ["Drop 'internal_id' before training."],
  "risks": ["Class imbalance in 'churn' may bias classifiers."],
  "raw_llm_response": null
}
```

Se o LLM retornar algo que não é JSON, `raw_llm_response` traz o texto bruto para debug e as listas ficam vazias — a rota **não** falha (segue com HTTP 201).

**Erros**:

- `404 dataset_not_found` — dataset inexistente.
- `400 unknown_llm_provider` — nome do provider não registrado.
- `400 missing_llm_credentials` — API key do provider ausente.
- `500 llm_error` — falha da SDK do LLM.

Exemplo:

```bash
curl -X POST http://localhost:8000/datasets/<dataset_id>/insights \
  -H "content-type: application/json" \
  -d '{}'
```

### Camada de LLMs

O módulo `app/llms/` fornece uma abstração comum sobre três provedores (OpenAI, Anthropic e Google Gemini). A intenção é que agentes e serviços futuros dependam desta interface — nunca dos SDKs diretamente.

**Estrutura**:

```text
app/llms/
    base.py                 LLMProvider (ABC) + Message + LLMResponse
    openai_provider.py      OpenAIProvider (OpenAI Chat Completions)
    anthropic_provider.py   AnthropicProvider (Messages API)
    gemini_provider.py      GeminiProvider (google-generativeai)
    factory.py              get_llm_provider(name, settings)
```

**Interface**:

```python
from app.llms.base import Message
from app.llms.factory import get_llm_provider

provider = get_llm_provider()  # usa DEFAULT_LLM_PROVIDER
response = provider.chat(
    [
        Message(role="system", content="Reply in English."),
        Message(role="user", content="Hi."),
    ],
    model="gpt-4o-mini",  # opcional; usa <provider>_model do settings se omitido
)
print(response.content, response.usage)
```

**Configuração** (`.env`):

```text
DEFAULT_LLM_PROVIDER=openai

OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini

ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001

GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
```

**Erros**:

- `400 unknown_llm_provider` — nome não registrado no factory.
- `400 missing_llm_credentials` — provider requisitado mas API key ausente.

**Observações**:

- Cada provider armazena internamente o cliente da SDK e traduz as mensagens no formato que o SDK espera. Vendor types não vazam para fora.
- OpenAI e Anthropic aceitam `system` como role dedicada; Gemini não tem esse conceito, então a instrução system é concatenada na última mensagem `user`.
- Tokens são reportados via `LLMResponse.usage` quando o SDK expõe (OpenAI e Anthropic; Gemini fica `null` por enquanto).
- Streaming e embeddings ainda não estão implementados nesta etapa.

### Explicabilidade de modelos

O endpoint `GET /models/{model_id}/explain` recarrega o pipeline treinado, aplica-o novamente ao dataset original e produz:

- **Feature importance** — do próprio estimador:
  - Modelos tree-based (RandomForest, DecisionTree) → `feature_importances_`
  - Modelos lineares (Logistic/Linear Regression) → `|coef_|` (magnitude dos coeficientes)
- **SHAP values** — explicações locais agregadas em magnitude média absoluta por feature:
  - Tree-based → `shap.TreeExplainer` (rápido)
  - Lineares → `shap.LinearExplainer`
  - Fallback → `shap.KernelExplainer` com sample de 100 linhas de background
- **Gráfico SHAP** — summary plot renderizado com matplotlib e salvo em `storage/charts/models/{model_id}/shap_summary.png`, servido via prefixo estático já existente.
- **Top features** — visão consolidada combinando importância e SHAP em uma única lista (top 10).
- **Narrativa** — texto curto em inglês listando as principais variáveis do modelo.

Para classificação binária a SHAP é computada sobre a classe positiva; para multiclass, retornamos a média das magnitudes absolutas entre classes.

Exemplo:

```bash
curl http://localhost:8000/models/<model_id>/explain
```

Resposta (resumida):

```json
{
  "model_id": "...",
  "dataset_id": "...",
  "algorithm": "random_forest",
  "problem_type": "classification",
  "target_column": "churn",
  "feature_importance": [
    {"feature": "num__salary", "importance": 0.42},
    {"feature": "num__age", "importance": 0.31}
  ],
  "shap": {
    "mean_abs_values": [
      {"feature": "num__salary", "value": 0.35},
      {"feature": "num__age", "value": 0.22}
    ],
    "chart_url": "/static/charts/models/<model_id>/shap_summary.png"
  },
  "top_features": [
    {"feature": "num__salary", "importance": 0.42, "mean_abs_shap": 0.35}
  ],
  "narrative": "The model relies primarily on 'num__salary', 'num__age'..."
}
```

Nomes de features vêm com o prefixo aplicado pelo `ColumnTransformer` (`num__` para numéricas escalonadas, `cat__` para categóricas one-hot).

### Machine Learning

O endpoint `POST /datasets/{dataset_id}/train` treina um conjunto fixo de algoritmos sobre um dataset já enviado e persiste o pipeline vencedor.

**Request**:

```json
{
  "target_column": "churn",
  "problem_type": "classification",
  "test_size": 0.2,
  "cv_folds": 5
}
```

- `target_column`: nome exato de uma coluna existente no dataset.
- `problem_type`: `classification` ou `regression`.
- `test_size` (opcional): fração das linhas para teste (default `0.2`).
- `cv_folds` (opcional): folds da cross-validation (default `5`).

**Candidatos avaliados**:

| Tipo | Algoritmos |
|------|------------|
| `classification` | `logistic_regression`, `decision_tree`, `random_forest` |
| `regression` | `linear_regression`, `random_forest` |

**Pipeline aplicado** (via sklearn `Pipeline` + `ColumnTransformer`):

1. Feature selection: descarta colunas com >90% de nulos ou (para categóricas) mais de 50 valores únicos.
2. Numéricas: `SimpleImputer(median)` + `StandardScaler`.
3. Categóricas: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`.
4. Split treino/teste (estratificado quando classification).
5. Cross-validation em cada candidato usando `f1_weighted` (classification) ou `r2` (regression).
6. Fit final no treino e avaliação no test set.
7. Vencedor: candidato com maior média de score na CV.

**Métricas retornadas** (`ModelMetrics`):

- Classification: `accuracy`, `precision`, `recall`, `f1`, `roc_auc` (apenas binary com `predict_proba`).
- Regression: `r2`, `mae`, `rmse`.

Campos não aplicáveis vêm `null`.

**Persistência**: o pipeline completo (preprocessador + estimador) é salvo em `storage/models/{model_id}.joblib` acompanhado de um manifest JSON com metadados (dataset, target, features, algoritmo escolhido).

**Erros**:

- `400 invalid_target_column` — coluna alvo não existe no dataset.
- `422 insufficient_data` — menos de 10 linhas usáveis ou target de classificação com única classe.
- `404 dataset_not_found` — dataset id inexistente.
- `422 validation_error` — corpo inválido (ex.: `problem_type` fora do enum).

Exemplo:

```bash
curl -X POST http://localhost:8000/datasets/<dataset_id>/train \
  -H "content-type: application/json" \
  -d '{"target_column": "churn", "problem_type": "classification"}'
```

Resposta (resumida):

```json
{
  "dataset_id": "...",
  "model_id": "...",
  "problem_type": "classification",
  "target_column": "churn",
  "features": ["age", "salary", "city"],
  "chosen_algorithm": "random_forest",
  "n_samples_train": 120,
  "n_samples_test": 30,
  "candidates": [
    {
      "algorithm": "logistic_regression",
      "cv_score_mean": 0.87,
      "cv_score_std": 0.03,
      "test_metrics": {"accuracy": 0.83, "precision": 0.82, "recall": 0.81, "f1": 0.81, "roc_auc": 0.91}
    },
    {"algorithm": "decision_tree", "...": "..."},
    {"algorithm": "random_forest", "...": "..."}
  ],
  "best_metrics": {"accuracy": 0.90, "precision": 0.89, "recall": 0.90, "f1": 0.89, "roc_auc": 0.95},
  "model_uri": "storage/models/<model_id>.joblib"
}
```

### Visualizações automáticas

O endpoint `GET /datasets/{dataset_id}/charts` renderiza cinco tipos de gráficos exploratórios em **dois formatos por gráfico**:

- **PNG estático** com matplotlib (backend Agg, sem GUI).
- **HTML interativo** com plotly (zoom, hover, tooltips; `plotly.js` via CDN).

Os arquivos são gravados em `storage/charts/{dataset_id}/` e servidos como estáticos sob o prefixo configurável `CHARTS_STATIC_URL_PREFIX` (default `/static/charts`).

| Grupo | Origem | Formato |
|-------|--------|---------|
| `histograms` | Uma imagem por coluna numérica. | PNG + HTML |
| `boxplots` | Uma imagem por coluna numérica. | PNG + HTML |
| `correlation_heatmap` | Matriz de correlação de Pearson entre colunas numéricas (nulo se `<2` colunas numéricas). | PNG + HTML |
| `bar_charts` | Top 10 valores mais frequentes por coluna categórica. | PNG + HTML |
| `category_distributions` | Bar chart agregado com o número de valores únicos por coluna categórica (nulo se não houver colunas categóricas). | PNG + HTML |

Exemplo:

```bash
curl http://localhost:8000/datasets/<dataset_id>/charts
```

Resposta (resumida):

```json
{
  "dataset_id": "...",
  "charts": {
    "histograms": [
      {
        "column": "age",
        "png_url": "/static/charts/<id>/histogram_age.png",
        "html_url": "/static/charts/<id>/histogram_age.html"
      }
    ],
    "boxplots": [ ... ],
    "correlation_heatmap": {
      "png_url": "/static/charts/<id>/correlation_heatmap.png",
      "html_url": "/static/charts/<id>/correlation_heatmap.html"
    },
    "bar_charts": [ ... ],
    "category_distributions": {
      "png_url": "/static/charts/<id>/category_distribution.png",
      "html_url": "/static/charts/<id>/category_distribution.html"
    }
  }
}
```

Chamar o endpoint novamente para o mesmo dataset **sobrescreve** os arquivos existentes (operação idempotente).

### Data cleaning

O endpoint `POST /datasets/{dataset_id}/clean` executa um pipeline configurável e persiste o resultado como **um novo dataset** — o original nunca é alterado.

Passos aplicados (todos ativados por padrão; envie `false` no body para desligar):

| Flag | O que faz |
|------|-----------|
| `standardize_column_names` | Normaliza nomes de colunas para snake_case ASCII (remove acentos, espaços e caracteres especiais). |
| `strip_whitespace` | Remove espaços nas extremidades de colunas textuais; strings vazias viram nulos. |
| `remove_empty_rows` | Remove linhas onde todos os valores são nulos. |
| `remove_duplicates` | Remove linhas duplicadas. |
| `convert_types` | Converte automaticamente `object` para numérico ou datetime quando todos os valores não-nulos convertem. |
| `fill_nulls` | Preenche nulos: mediana para numéricas, moda para categóricas, primeiro valor observado para datetime; colunas 100% nulas são deixadas como estão. |

Exemplo:

```bash
# Aplica todo o pipeline (body vazio ou {})
curl -X POST http://localhost:8000/datasets/<dataset_id>/clean \
  -H "content-type: application/json" \
  -d '{}'

# Apenas remove duplicados
curl -X POST http://localhost:8000/datasets/<dataset_id>/clean \
  -H "content-type: application/json" \
  -d '{
    "remove_duplicates": true,
    "remove_empty_rows": false,
    "fill_nulls": false,
    "strip_whitespace": false,
    "standardize_column_names": false,
    "convert_types": false
  }'
```

Resposta:

```json
{
  "original_dataset_id": "...",
  "cleaned_dataset_id": "...",
  "report": {
    "rows_before": 1250,
    "rows_after": 1220,
    "rows_removed": 30,
    "duplicates_removed": 25,
    "empty_rows_removed": 5,
    "nulls_filled": {"age": 12, "city": 3},
    "whitespace_stripped_columns": ["name", "email"],
    "columns_renamed": {"Nome do Cliente": "nome_do_cliente"},
    "types_converted": {"age": {"before": "object", "after": "int64"}},
    "operations_applied": [
      "standardize_column_names", "strip_whitespace",
      "remove_empty_rows", "remove_duplicates",
      "convert_types", "fill_nulls"
    ]
  }
}
```

O `cleaned_dataset_id` retornado pode ser consultado pelos endpoints de summary como qualquer outro dataset.

### Análise exploratória (EDA)

O endpoint `GET /datasets/{dataset_id}/summary` gera um resumo estatístico completo do dataset já enviado:

- **Estrutura**: quantidade de linhas, colunas, uso de memória (string legível) e número de linhas duplicadas.
- **Nulos**: contagem absoluta e percentual (0-100) por coluna.
- **Tipos**: dtype Pandas por coluna, além das listas de colunas numéricas e categóricas.
- **Estatísticas numéricas**: média, mediana, mínimo, máximo, desvio padrão (amostral, ddof=1) e quartis (Q1, Q2, Q3).
- **Estatísticas categóricas**: número de valores únicos e os top 5 valores mais frequentes.

Estatísticas ficam totalmente encapsuladas em `EDAService` (`app/services/eda_service.py`); a rota apenas delega. O CSV é lido do disco a cada chamada — sem cache neste momento.

## Qualidade

```bash
# Lint
uv run ruff check .

# Formatação (verificação)
uv run ruff format --check .
uv run black --check .

# Formatação (aplicar)
uv run ruff format .
uv run black .

# Testes
uv run pytest
```

## CI

O workflow em `.github/workflows/ci.yml` executa lint, verificação de formatação e testes em cada push e pull request para `main`.

## Convenções

- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) (`feat`, `fix`, `chore`, `build`, `ci`, `docs`, `test`, `refactor`).
- **Idioma**: código, docstrings e mensagens de commit em inglês; documentação de projeto em português.
- **Branch principal**: `main`.

## Roadmap

Etapas concluídas:

1. **Etapa 1** — scaffolding da arquitetura, configuração base, logging, CI.
2. **Etapa 2** — API FastAPI com endpoints `/`, `/health`, `/datasets/upload`, tratamento global de exceções, validação Pydantic e inspeção de CSV com Pandas (encoding, separador, tipos de coluna).
3. **Etapa 3** — análise exploratória de dados (EDA) via `GET /datasets/{id}/summary`, com estatísticas descritivas para colunas numéricas e categóricas, contagem de nulos e duplicados.
4. **Etapa 4** — data cleaning configurável via `POST /datasets/{id}/clean`: dedup, remoção de linhas vazias, strip de whitespace, padronização de nomes de coluna, conversão automática de tipos e preenchimento de nulos; salva o resultado como um novo dataset.
5. **Etapa 5** — geração automática de visualizações via `GET /datasets/{id}/charts`: histogramas, boxplots, heatmap de correlação, gráficos de barras e distribuição de categorias, em PNG (matplotlib) e HTML interativo (plotly), servidos como estáticos.
6. **Etapa 6** — pipeline de Machine Learning via `POST /datasets/{id}/train`: classificação (LogisticRegression, DecisionTree, RandomForest) e regressão (LinearRegression, RandomForestRegressor), com pré-processamento automático (imputação, encoding, escalonamento), cross-validation, seleção do vencedor e persistência do pipeline treinado em joblib.
7. **Etapa 7** — explicabilidade via `GET /models/{id}/explain`: feature importance do estimador, SHAP values com explainer escolhido pelo tipo do modelo, summary plot em PNG e narrativa curta.
8. **Etapa 8** — camada de abstração de LLMs em `app/llms/`: interface comum (`LLMProvider`), providers para OpenAI, Anthropic e Google Gemini, factory por nome, configuração via `.env`, sem endpoint HTTP nesta etapa.
9. **Etapa 9** — insights gerados por LLM via `POST /datasets/{id}/insights`: EDA feed ao modelo, resposta estruturada com resumo executivo, insights, anomalias, sugestões e riscos; fallback para `raw_llm_response` quando o modelo não devolve JSON.
10. **Etapa 10** — agente LangChain via `POST /agent/chat`: cinco ferramentas (`dataset_info`, `dataset_summary`, `column_statistics`, `train_model`, `generate_charts`) e um agente `create_agent` (LangChain 1.x) sobre `ChatOpenAI` capaz de encadear as chamadas para responder em linguagem natural.

Próximas etapas planejadas:

11. Primeiro pipeline de análise ponta a ponta.
12. Persistência estruturada e camada de repositório sobre banco de dados.

## Licença

MIT.
