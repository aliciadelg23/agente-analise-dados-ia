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

Próximas etapas planejadas:

5. Configuração de provedores de LLM em `app/llms/`.
6. Definição de contratos base para agentes em `app/agents/`.
7. Primeiro pipeline de análise ponta a ponta.
8. Persistência estruturada e camada de repositório sobre banco de dados.

## Licença

MIT.
