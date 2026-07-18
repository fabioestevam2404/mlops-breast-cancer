# ARCHITECTURE.md

## Visão Geral

```
sklearn dataset ──► data.py (split + scaling em Pipeline)
                        │
                        ▼
        train.py ── LogReg (baseline) ─┐
                 └─ RandomForest ──────┤──► MLflow Tracking (sqlite:///mlflow.db)
                        │              │         │
                        ▼              │         ▼
        tune.py ── RandomizedSearchCV ─┘   Model Registry
                └─ HalvingGridSearchCV      (alias: champion)
                        │                        │
                        ▼                        ▼
        drift.py ◄─── monitoramento     models/model.joblib (export p/ imagem)
                                                 │
                                                 ▼
                                     api/main.py (FastAPI /predict)
                                                 │
                                          Dockerfile (multi-stage)
                                                 │
                                     k8s/deployment.yaml (2 réplicas)
                                     k8s/service.yaml (LoadBalancer)
```

## Decisões-chave
- **Pipeline sklearn (scaler + modelo) como artefato único**: elimina training-serving skew.
- **SQLite como backend do MLflow**: único modo local que habilita Model Registry.
- **Export do modelo para models/model.joblib no build**: a imagem não depende
  do MLflow server em runtime (12-factor: dependências explícitas, sem estado externo).
- **Probes**: /health para liveness e readiness — o pod só recebe tráfego com modelo carregado.

## Fluxo de Predição (runtime)
1. Pod inicia → lifespan carrega model.joblib em memória
2. POST /predict → Pydantic valida 30 features → pipeline.predict_proba
3. Resposta: classe, probabilidade, versão do modelo, latência
