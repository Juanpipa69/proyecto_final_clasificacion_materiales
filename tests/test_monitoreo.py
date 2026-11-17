"""Pruebas del monitoreo de deriva: no debe alertar con datos similares y
si debe alertar con materiales de un dominio distinto."""
import joblib

from Monitoring.drift_monitoring import comparar, perfil
from src.models.predict import Clasificador

FUERA_DE_DOMINIO = [
    "licencia anual software contable multiusuario",
    "servicio de consultoria juridica externa",
    "pasaje aereo nacional ida y regreso",
    "suscripcion revista digital mensual",
    "capacitacion virtual liderazgo equipos",
] * 20


def _modelo(modelo_path):
    b = joblib.load(modelo_path)
    return b["pipeline"], b["classes"]


def test_sin_alerta_con_datos_similares(modelo_path, df):
    model, clases = _modelo(modelo_path)
    ref = perfil(model, df.iloc[:500], clases)
    nuevo = perfil(model, df.iloc[500:1000], clases)
    assert comparar(ref, nuevo, clases)["alertas"] == []


def test_alerta_con_materiales_fuera_de_dominio(modelo_path, df):
    model, clases = _modelo(modelo_path)
    ref = perfil(model, df.iloc[:500], clases)
    lote = Clasificador._preparar([{"texto_breve": t} for t in FUERA_DE_DOMINIO])
    resultado = comparar(ref, perfil(model, lote, clases), clases)
    assert resultado["alertas"], resultado
    assert resultado["recomendacion"] == "Revisar y reentrenar"
