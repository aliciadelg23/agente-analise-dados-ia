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

A API sobe em `http://localhost:8000`. Endpoint de verificação:

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"0.1.0"}
```

Documentação interativa gerada pelo FastAPI:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

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

Esta é a etapa 1 (scaffolding). Próximas etapas planejadas:

1. Configuração de provedores de LLM em `app/llms/`.
2. Definição de contratos base para agentes em `app/agents/`.
3. Primeiro pipeline de análise ponta a ponta.
4. Persistência e camada de repositório.

## Licença

MIT.
