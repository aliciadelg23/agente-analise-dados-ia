# Arquitetura

Este documento resume as decisões arquiteturais desta etapa inicial. Detalhes serão expandidos conforme novas funcionalidades forem implementadas.

## Camadas

| Camada | Pasta | Responsabilidade |
|--------|-------|------------------|
| Entrada HTTP | `app/api` | Routers, dependências e schemas de request/response. |
| Configuração | `app/config` | Carga e validação de variáveis de ambiente. |
| Core | `app/core` | Preocupações transversais (logging, exceções, segurança). |
| Modelos | `app/models` | Modelos de domínio e schemas Pydantic reutilizáveis. |
| Serviços | `app/services` | Regras de negócio e coordenação entre agentes e repositórios. |
| Repositórios | `app/repositories` | Abstrações sobre persistência. |
| Agentes | `app/agents` | Agentes de IA especializados. |
| Pipelines | `app/pipelines` | Pipelines de processamento composíveis. |
| LLMs | `app/llms` | Adaptadores para provedores de LLM. |
| Utilitários | `app/utils` | Funções auxiliares sem dependências internas. |

## Princípios adotados

- **Separação de responsabilidades**: cada pasta tem um único motivo para mudar.
- **Inversão de dependências**: camadas de alto nível (serviços, agentes) dependem de interfaces expostas por camadas de infraestrutura (repositórios, LLMs), nunca do contrário.
- **Fábrica de aplicação**: `create_app()` monta a instância FastAPI, o que facilita testes e alternativas de deploy.
- **Configuração tipada**: `pydantic-settings` valida variáveis de ambiente na inicialização, evitando erros só percebidos em runtime.

## Fluxo de uma requisição futura (referência)

```text
HTTP  →  app/api/routes  →  app/services  →  app/agents / app/pipelines
                                  ↓
                          app/repositories  →  storage
```

## Tratamento de erros

- Exceções de domínio herdam de `app.core.exceptions.AppError`, que carrega `status_code` e `code`.
- `app.core.exception_handlers.register_exception_handlers()` converte `AppError`, `RequestValidationError` e `Exception` em respostas com o formato:

```json
{ "error": { "code": "invalid_file_extension", "message": "Unsupported file extension." } }
```

- Erros esperados são logados como `warning`; erros inesperados como `exception` (com traceback).

## Fluxo do upload de datasets (Etapa 2)

```text
POST /datasets/upload
  → app/api/routes/datasets.py       parse UploadFile
  → app/services/DatasetService      valida extensão, tamanho, não-vazio
      → app/repositories/DatasetRepository   persiste em disco
      → app/utils/csv_inspector              detecta encoding/sep/tipos
  → DatasetUploadResponse (Pydantic)
```

## Onde adicionar coisas novas

- Novo endpoint: criar um módulo em `app/api/routes/` e registrar no `create_app()`.
- Nova regra de negócio: criar um serviço em `app/services/`.
- Novo agente: criar uma classe em `app/agents/` implementando o contrato base (a ser definido na próxima etapa).
- Novo provedor de LLM: criar um adaptador em `app/llms/` respeitando a interface comum.
