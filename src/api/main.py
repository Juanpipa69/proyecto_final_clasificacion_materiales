"""API REST para clasificar materiales por familia.

Endpoints:
  GET  /health      estado del servicio y del modelo (sin autenticacion)
  GET  /model-info  algoritmo, clases y fecha de entrenamiento
  POST /predict     clasifica de 1 a API_MAX_ITEMS materiales por solicitud

Variables de entorno:
  MODEL_PATH     ruta del modelo (por defecto models/champion.joblib)
  API_KEY        si se define, /model-info y /predict exigen el header X-API-Key
  API_MAX_ITEMS  maximo de materiales por solicitud (por defecto 100)

Por privacidad, los registros (logs) solo guardan cuantos materiales llegaron y
cuanto tardo la respuesta; nunca el texto de los materiales.
Documentacion interactiva: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import os
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import Clasificador

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MAX_ITEMS = int(os.getenv("API_MAX_ITEMS", "100"))
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una sola vez al iniciar el servicio."""
    try:
        _state["clf"] = Clasificador(os.getenv("MODEL_PATH") or None)
        logger.info("Modelo cargado: %s", _state["clf"].info)
    except Exception as exc:  # el servicio arranca igual y /health informa el problema
        _state["error"] = str(exc)
        logger.error("No se pudo cargar el modelo: %s", exc)
    yield
    _state.clear()


app = FastAPI(
    title="Clasificador de materiales",
    description="Predice la familia de un material a partir de su descripcion.",
    version="1.0.0",
    lifespan=lifespan,
)


class Material(BaseModel):
    texto_breve: str = Field(..., min_length=3, max_length=200,
                             examples=["GUANTE DE NITRILO TALLA L"])
    texto_largo: str = Field("", max_length=1000)
    marca: str = Field("SIN_MARCA", max_length=60)
    um: str = Field("", max_length=10, examples=["par"])
    precio_usd: float = Field(0.0, ge=0, le=10_000_000)
    filas_me2m: int = Field(0, ge=0, le=1_000_000)
    items_ariba: int = Field(0, ge=0, le=1_000_000)


class SolicitudPrediccion(BaseModel):
    materiales: list[Material] = Field(..., min_length=1, max_length=MAX_ITEMS)


class Alternativa(BaseModel):
    familia: str
    probabilidad: float


class Prediccion(BaseModel):
    familia: str
    confianza: float
    alternativas: list[Alternativa]


class RespuestaPrediccion(BaseModel):
    modelo: dict
    predicciones: list[Prediccion]


def verificar_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Si API_KEY esta definida, exige el header X-API-Key con ese valor."""
    esperada = os.getenv("API_KEY")
    if esperada and not (x_api_key and secrets.compare_digest(x_api_key, esperada)):
        raise HTTPException(status_code=401, detail="API key invalida o ausente")


def _clasificador() -> Clasificador:
    if "clf" not in _state:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    return _state["clf"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if "clf" in _state else "sin_modelo",
            "modelo_cargado": "clf" in _state}


@app.get("/model-info", dependencies=[Depends(verificar_api_key)])
def model_info() -> dict:
    clf = _clasificador()
    return {**clf.info, "clases": clf.classes, "max_items_por_solicitud": MAX_ITEMS}


@app.post("/predict", response_model=RespuestaPrediccion,
          dependencies=[Depends(verificar_api_key)])
def predict(solicitud: SolicitudPrediccion) -> RespuestaPrediccion:
    clf = _clasificador()
    t0 = time.perf_counter()
    try:
        preds = clf.predecir([m.model_dump() for m in solicitud.materiales])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info("predict: %d materiales en %.1f ms",
                len(solicitud.materiales), (time.perf_counter() - t0) * 1000)
    return RespuestaPrediccion(modelo=clf.info, predicciones=preds)
