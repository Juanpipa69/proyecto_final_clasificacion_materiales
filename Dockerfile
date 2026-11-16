# Imagen unica para entrenar, servir MLflow y exponer la API.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: libreria de calculo paralelo que necesita XGBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias primero (se cachean mientras requirements.txt no cambie)
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY configs/ configs/
COPY src/ src/

# Usuario sin privilegios; carpetas de escritura creadas con su propietario
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/models /app/reports /mlflow \
    && chown -R appuser:appuser /app /mlflow
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
