# PRD — Pipeline MLOps: Breast Cancer Classifier em Produção

## 1. Problema
Demonstrar, em projeto de portfólio, o ciclo de vida completo de um modelo de ML:
da análise exploratória ao deploy escalável e monitorado em Kubernetes.

## 2. Objetivos
| # | Objetivo | Critério de aceite |
|---|----------|--------------------|
| O1 | EDA e pré-processamento reprodutíveis | Notebook com estatísticas, histogramas, correlação; split estratificado 80/20 |
| O2 | Baseline honesto | Logistic Regression comparada ao Random Forest com 5 métricas + matriz de confusão |
| O3 | Tuning rastreado | RandomizedSearchCV e HalvingGridSearchCV logados no MLflow; melhor modelo no Registry |
| O4 | Monitoramento | Função de simulação de drift + métricas de produção logadas no MLflow |
| O5 | Serving | Endpoint /predict (FastAPI) com validação de schema e /health |
| O6 | Containerização | docker build + docker run funcionais; imagem < 600 MB |
| O7 | Orquestração | Deployment com 2 réplicas + Service; rollout testado; probes configurados |

## 3. Não-objetivos (v1)
- Retraining automático agendado (apenas simulado no relatório)
- Autenticação na API
- GPU / deep learning

## 4. Métricas de Sucesso do Modelo
- Recall (classe maligna) ≥ 0.95 no teste — métrica primária (ADR-004)
- F1 ≥ 0.95 · ROC-AUC ≥ 0.99 (dataset é sabidamente separável)

## 5. Usuários
- Recrutadores/avaliadores técnicos (portfólio)
- O próprio autor, como template para projetos futuros de MLOps
