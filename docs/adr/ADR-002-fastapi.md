# ADR-002 — FastAPI em vez de Flask

**Status**: Aceita · **Data**: 2026-07-17

## Contexto
O enunciado permite Flask ou FastAPI para o endpoint /predict.

## Decisão
FastAPI + Pydantic v2 + Uvicorn.

## Justificativa
1. Validação de schema nativa: as 30 features do payload são validadas automaticamente
   (tipo, presença, faixa) — em Flask isso seria código manual propenso a erro.
2. Documentação OpenAPI gerada de graça em /docs — ótimo para demo de portfólio.
3. Async e performance superiores sob carga (relevante no teste de escala do K8s).
4. Alinhamento com o restante do portfólio (LLM+RAG scaffolding já usa FastAPI).

## Consequências
− Curva de aprendizado de Pydantic v2 (mitigada: schemas simples)
