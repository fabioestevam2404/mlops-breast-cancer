# ---------- Estágio 1: builder (instala dependências em venv isolada) ----------
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements-api.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -U pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements-api.txt

# ---------- Estágio 2: runtime (imagem final enxuta) ----------
FROM python:3.12-slim

# Usuário não-root (boa prática de segurança / Bandit B*)
RUN useradd --create-home appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=/app/models/model.joblib \
    PYTHONUNBUFFERED=1

COPY api/ ./api/
COPY models/model.joblib models/MODEL_VERSION ./models/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen(\"http://localhost:8000/health\")"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
