# Posts LinkedIn — 3 variantes

## Variante 1 — Técnica (audiência: data/ML engineers)

🚀 Acabei de finalizar um pipeline de MLOps de ponta a ponta — e o maior aprendizado não foi o modelo.

O projeto: classificador para o Breast Cancer Wisconsin, cobrindo o ciclo completo:

🔹 EDA + split estratificado com scaler DENTRO do Pipeline sklearn (zero leakage, zero training-serving skew)
🔹 RandomizedSearchCV vs HalvingGridSearchCV com scoring=recall — porque no domínio médico, falso negativo é o erro que custa caro
🔹 MLflow com backend SQLite: tracking + Model Registry com alias "champion"
🔹 Monitoramento com PSI: o índice cruzou o limiar de alerta DUAS semanas antes do F1 colapsar no drift simulado
🔹 FastAPI + Docker multi-stage (non-root) + Kubernetes com 2 réplicas, probes e rolling update zero-downtime

O insight que levo: monitorar só a métrica-alvo esconde o problema. O PSI é o detector de fumaça; o F1 é o prédio já pegando fogo.

16 testes, 97.7% de cobertura, CI com lint + Bandit + smoke test do container.

Código no GitHub: [link]

#MLOps #MachineLearning #MLflow #Kubernetes #DataEngineering #Python

## Variante 2 — Ampla (audiência: geral/recrutadores)

Um modelo de machine learning que só funciona no notebook do cientista de dados não gera valor nenhum. 📉

Foi por isso que meu último projeto de portfólio não parou no treino do modelo — fui até o deploy:

✅ Modelo de classificação com 97% de acurácia
✅ Cada experimento rastreado e versionado (nada de "modelo_final_v3_AGORA_VAI.pkl")
✅ Sistema de alerta que detecta quando os dados de produção mudam — antes de o modelo errar
✅ API rodando em containers com 2 réplicas: se uma cai, a outra segura o tráfego

A parte mais interessante? Simular o "envelhecimento" do modelo em produção e ver o sistema de monitoramento acusar o problema semanas antes das métricas caírem.

É a diferença entre saber treinar um modelo e saber operar um modelo. 🎯

#MachineLearning #MLOps #Tecnologia #Dados #Python

## Variante 3 — Metodologia (audiência: tech leads / Tech Leads Club)

Spec-Driven Development aplicado a MLOps: o overhead que se paga em uma tarde. 🧵

No meu último projeto (pipeline completo: sklearn → MLflow → K8s), antes de escrever qualquer código de modelo, escrevi:

📋 PRD com critérios de aceite mensuráveis por objetivo
🏛️ 4 ADRs — incluindo o mais importante: POR QUE recall é a métrica de tuning (falso negativo em diagnóstico de câncer é o erro mais caro)
🗂️ 8 TASK files atômicos com Definition of Done
🤖 CLAUDE.md travando as convenções para AI-assisted development

O resultado prático: quando o tuning terminou com recall abaixo da meta do PRD, a decisão do que fazer já estava tomada — o ADR-004 define o trade-off aceitável e o PRD registra o caminho (calibrar threshold na v1.1). Sem debate, sem retrabalho, sem "deixa eu pensar de novo do zero".

Documentação de decisão não é burocracia. É cache de raciocínio. ⚡

#TechLeadership #SpecDriven #MLOps #EngenhariaDeSoftware #ADR
