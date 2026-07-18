# 🩺 MLOps End-to-End — Breast Cancer Classifier

> Ciclo de vida completo de um modelo de ML: **EDA → treino → tuning → MLflow
> Registry → monitoramento de drift → FastAPI → Docker → Kubernetes**.
> Projeto de portfólio com metodologia Spec-Driven Development.

![CI](https://github.com/fabioestevam2404/mlops-breast-cancer/actions/workflows/ci.yml/badge.svg)

## Arquitetura

```
sklearn dataset ─► data.py ─► train.py (LogReg vs RF) ─► MLflow Tracking (SQLite)
                                  │                            │
                                  ▼                            ▼
                    tune.py (Randomized + HalvingGrid) ─► Model Registry @champion
                                  │                            │
                                  ▼                            ▼
                    drift.py (PSI + métricas de produção)   models/model.joblib
                                                               │
                                            api/main.py (FastAPI /predict, /health)
                                                               │
                                            Dockerfile multi-stage (non-root)
                                                               │
                                     k8s/ (Deployment 2 réplicas + LoadBalancer)
```

## Resultados

| Modelo | Recall (maligno) | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| LogReg (baseline) | 0.929 | 0.975 | 0.951 | 0.996 |
| RF default | 0.929 | 1.000 | 0.963 | 0.993 |
| **RF tunado (champion)** | **0.929** | 0.975 | 0.951 | 0.995 |

Métrica primária: **recall da classe maligna** (falso negativo = câncer não
detectado é o erro mais caro — ver `docs/adr/ADR-004`).

Monitoramento: o **PSI cruza o limiar de 0.25 duas semanas antes** de o F1
colapsar sob drift simulado — o alarme antecipado que dispararia retraining.

## Quickstart

```bash
make setup                 # venv + deps
source .venv/bin/activate
make data train tune       # pipeline completo, tudo logado no MLflow
make drift                 # simulação de monitoramento
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
make api                   # http://localhost:8000/docs
```

### Docker + Kubernetes (Docker Desktop)

```bash
make docker-build && make docker-run     # teste local do container
kubectl apply -f k8s/                    # 2 réplicas + LoadBalancer
kubectl get svc breast-cancer-api        # EXTERNAL-IP: localhost
curl localhost:8000/health
kubectl scale deployment breast-cancer-api --replicas=4   # teste de escala
```

### Exemplo de predição

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d @docs/exemplo_payload.json
# {"prediction":1,"diagnosis":"maligno","probability_malignant":0.866,...}
```

## Qualidade

- **16 testes · 97.7% de cobertura** (pytest + pytest-cov, gate ≥ 80%)
- Ruff (lint + format) · mypy · Bandit no CI
- GitHub Actions: quality gate + build da imagem + smoke test do container
- Spec-Driven: `CLAUDE.md`, `docs/PRD.md`, `docs/ARCHITECTURE.md`, 4 ADRs,
  8 TASK files atômicos em `.claude/tasks/`

## Estrutura

```
├── CLAUDE.md               # contexto para AI-assisted development
├── docs/                   # PRD, ARCHITECTURE, ADRs, posts LinkedIn
├── src/mlops_bc/           # config, data, train, tune, drift
├── api/                    # FastAPI (main + schemas Pydantic)
├── tests/                  # 16 testes
├── notebooks/              # relatório final executado (ipynb)
├── k8s/                    # deployment.yaml + service.yaml
├── Dockerfile              # multi-stage, non-root, HEALTHCHECK
└── .github/workflows/      # CI
```
