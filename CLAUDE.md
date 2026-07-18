# CLAUDE.md — Projeto MLOps Breast Cancer (Ciclo de Vida Completo)

> Este arquivo é lido automaticamente pelo Claude Code no início de cada sessão.
> Ele evita re-briefing e trava as decisões arquiteturais do projeto.

## Contexto do Projeto

Pipeline MLOps end-to-end sobre o dataset **Breast Cancer Wisconsin** (scikit-learn):
dados → pré-processamento → modelagem (LogReg baseline vs Random Forest) → tuning
(RandomizedSearchCV + HalvingGridSearchCV) → MLflow Tracking + Model Registry →
monitoramento de drift → API FastAPI `/predict` → Docker → Kubernetes (Docker Desktop).

## Stack Travada (ver ADRs em docs/adr/)

- Python 3.12 · scikit-learn · MLflow (tracking local em `./mlruns`)
- FastAPI + Pydantic v2 + Uvicorn (ADR-002: FastAPI > Flask)
- Docker multi-stage · Kubernetes do Docker Desktop (ADR-003)
- Qualidade: Ruff (lint+format), mypy, pytest (cobertura ≥ 80%)

## Comandos Principais

```bash
make setup        # venv + dependências
make data         # gera dados processados em data/processed/
make train        # baseline + random forest, loga no MLflow
make tune         # RandomizedSearch + HalvingGridSearch, registra melhor modelo
make drift        # simulação de data drift + log de métricas de produção
make api          # sobe FastAPI local (uvicorn)
make test         # pytest + ruff + mypy
make docker-build # build da imagem
make k8s-deploy   # kubectl apply -f k8s/
mlflow ui         # UI do MLflow em http://localhost:5000
```

## Convenções Obrigatórias

1. Todo experimento DEVE ser logado no MLflow (params, métricas, artefatos).
2. O modelo servido pela API vem SEMPRE do Model Registry (alias `champion`),
   com fallback para `models/model.joblib` dentro do container.
3. Métricas mínimas reportadas: accuracy, precision, recall, f1, ROC-AUC + matriz de confusão.
4. `recall` é a métrica de decisão para tuning (falso negativo = câncer não detectado
   é o erro mais caro no domínio). ADR-004.
5. Scaler e modelo são versionados JUNTOS (Pipeline sklearn) — nunca separados.
6. Nenhum caminho absoluto no código; usar `src/mlops_bc/config.py`.

## Armadilhas Conhecidas

- `HalvingGridSearchCV` é experimental: exige `from sklearn.experimental import enable_halving_search_cv` ANTES do import.
- MLflow Model Registry local exige backend SQLite: `mlflow.set_tracking_uri("sqlite:///mlflow.db")` — file store puro NÃO suporta registry.
- No K8s do Docker Desktop, `LoadBalancer` funciona e expõe em `localhost` (vpnkit). NodePort é o fallback.
- A imagem Docker NÃO deve conter o MLflow server — apenas o artefato do modelo copiado no build.
