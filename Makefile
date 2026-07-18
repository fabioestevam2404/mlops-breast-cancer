.PHONY: setup data train tune drift api test docker-build docker-run k8s-deploy k8s-destroy

PY=python

setup:
	python -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements-dev.txt

data:
	$(PY) -m mlops_bc.data

train:
	$(PY) -m mlops_bc.train

tune:
	$(PY) -m mlops_bc.tune

drift:
	$(PY) -m mlops_bc.drift

api:
	uvicorn api.main:app --reload --port 8000

test:
	ruff check src api tests && mypy src api && pytest

docker-build:
	docker build -t breast-cancer-api:1.0.0 .

docker-run:
	docker run --rm -p 8000:8000 breast-cancer-api:1.0.0

k8s-deploy:
	kubectl apply -f k8s/

k8s-destroy:
	kubectl delete -f k8s/
