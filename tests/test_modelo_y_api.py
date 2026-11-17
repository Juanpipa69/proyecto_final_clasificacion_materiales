"""Pruebas de inferencia y de la API (validacion, seguridad, respuestas)."""
import pytest
from fastapi.testclient import TestClient

from src.models.predict import Clasificador

MATERIALES = [{"texto_breve": "GUANTE DE NITRILO TALLA L", "um": "par"},
              {"texto_breve": "Cable THHN 12 AWG cobre"}]


def test_clasificador_devuelve_clases_validas(modelo_path):
    clf = Clasificador(modelo_path)
    preds = clf.predecir(MATERIALES)
    assert len(preds) == 2
    for p in preds:
        assert p["familia"] in clf.classes
        assert 0 <= p["confianza"] <= 1
        assert len(p["alternativas"]) == 2
    assert preds[0]["familia"] == "EPP"


def test_clasificador_rechaza_texto_vacio(modelo_path):
    with pytest.raises(ValueError):
        Clasificador(modelo_path).predecir([{"texto_breve": "   "}])


def test_clasificador_sin_modelo_da_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError):
        Clasificador(tmp_path / "no_existe.joblib")


@pytest.fixture()
def cliente(modelo_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(modelo_path))
    monkeypatch.setenv("API_KEY", "clave-de-prueba")
    from src.api.main import app
    with TestClient(app) as c:
        yield c


CLAVE = {"X-API-Key": "clave-de-prueba"}


def test_health_no_requiere_clave(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200 and r.json()["modelo_cargado"] is True


def test_predict_sin_clave_es_rechazado(cliente):
    r = cliente.post("/predict", json={"materiales": MATERIALES})
    assert r.status_code == 401


def test_predict_con_clave_incorrecta_es_rechazado(cliente):
    r = cliente.post("/predict", json={"materiales": MATERIALES}, headers={"X-API-Key": "x"})
    assert r.status_code == 401


def test_predict_con_clave_funciona(cliente):
    r = cliente.post("/predict", json={"materiales": MATERIALES}, headers=CLAVE)
    assert r.status_code == 200
    assert [p["familia"] for p in r.json()["predicciones"]][0] == "EPP"


@pytest.mark.parametrize("cuerpo", [
    {"materiales": []},                                         # lista vacia
    {"materiales": [{"texto_breve": ""}]},                      # texto vacio
    {"materiales": [{"texto_breve": "guante", "precio_usd": -1}]},  # precio negativo
    {"materiales": [{"texto_breve": "guante"}] * 101},          # mas del limite
    {"otro_campo": 1},                                          # formato incorrecto
])
def test_predict_rechaza_entradas_invalidas(cliente, cuerpo):
    assert cliente.post("/predict", json=cuerpo, headers=CLAVE).status_code == 422


def test_logs_no_guardan_el_texto(cliente, caplog):
    with caplog.at_level("INFO"):
        cliente.post("/predict", json={"materiales": MATERIALES}, headers=CLAVE)
    assert "NITRILO" not in caplog.text.upper()
    assert "materiales en" in caplog.text
