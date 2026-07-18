# ADR-004 — Recall como métrica primária de tuning

**Status**: Aceita · **Data**: 2026-07-17

## Contexto
Classificação binária maligno/benigno. Os dois erros têm custos assimétricos:
- Falso positivo: paciente saudável faz exames extras (custo baixo)
- Falso negativo: câncer não detectado (custo altíssimo)

## Decisão
`scoring="recall"` (classe maligna) nos buscadores de hiperparâmetros;
accuracy/precision/f1/ROC-AUC reportadas como métricas secundárias.

## Consequências
+ O modelo otimiza para o erro que importa no domínio
− Recall puro pode inflar falsos positivos; monitorar precision ≥ 0.90 como guarda
