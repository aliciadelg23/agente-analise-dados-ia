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

Próximas etapas planejadas:

3. Configuração de provedores de LLM em `app/llms/`.
4. Definição de contratos base para agentes em `app/agents/`.
5. Primeiro pipeline de análise ponta a ponta.
6. Persistência estruturada e camada de repositório sobre banco de dados.

## Licença

MIT.
