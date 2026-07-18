# ADR-003 — Kubernetes do Docker Desktop + Service LoadBalancer

**Status**: Aceita · **Data**: 2026-07-17

## Contexto
O enunciado pede Service tipo LoadBalancer OU NodePort. Ambiente escolhido:
Docker Desktop com K8s embutido (single-node).

## Decisão
Service **LoadBalancer** como padrão, NodePort documentado como fallback.

## Justificativa
No Docker Desktop, o vpnkit-controller provisiona LoadBalancer automaticamente
em localhost:<porta> — comportamento idêntico ao de um cloud provider, sem addons.
Em minikube seria necessário `minikube tunnel` (mais atrito).

## Consequências
+ `kubectl get svc` mostra EXTERNAL-IP=localhost; teste direto no navegador
− Em cluster real, trocar apenas o manifesto (anotações do cloud provider)
