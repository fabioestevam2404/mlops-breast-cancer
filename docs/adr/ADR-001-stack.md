# ADR-001 — Stack local: scikit-learn + MLflow + Docker Desktop

**Status**: Aceita · **Data**: 2026-07-17

## Contexto
Projeto didático de MLOps precisa rodar 100% na máquina local, sem custos de cloud,
mas usando as mesmas ferramentas de mercado.

## Decisão
scikit-learn (modelagem), MLflow com backend SQLite (tracking + registry),
Docker Desktop com Kubernetes embutido (orquestração).

## Consequências
+ Zero custo, setup em minutos, mesmas APIs usadas em produção real
+ K8s do Docker Desktop expõe LoadBalancer em localhost — sem MetalLB/minikube tunnel
− SQLite não suporta acesso concorrente de múltiplos usuários (aceitável: single-user)
− Sem GPU (irrelevante para Random Forest)
