"""Monitoreo de deriva (drift) del modelo en produccion.

Compara un lote nuevo de materiales contra la referencia (el test reservado,
datos que el modelo no vio al entrenar) con tres indicadores:

1. Palabras desconocidas (OOV): % de palabras del lote que no estan en el
   vocabulario del modelo. Si sube, los textos cambiaron de estilo o de tema.
2. Confianza media: si baja frente a la referencia, el modelo duda mas.
3. Distribucion de familias predichas (PSI): si cambia mucho, cambio la mezcla
   de materiales que llegan. PSI < 0.1 estable, 0.1-0.25 moderado, > 0.25 alto.

Si algun indicador supera su umbral, se recomienda revisar y reentrenar.
La demostracion usa como "lote nuevo" los 249 items de Ariba en texto libre,
que tienen una redaccion distinta a la del maestro de materiales.

Uso: python -m Monitoring.drift_monitoring
Salida: reports/monitoreo_drift.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import load_maestro
from src.evaluation.evaluate import load_external
from src.evaluation.metrics import load_champion
from src.models.train import TARGET, make_splits
from src.utils.config_loader import load_config

UMBRALES = {"oov": 0.25, "caida_confianza": 0.10, "psi": 0.25}


def tasa_oov(model, textos) -> float:
    """Proporcion de palabras que el vectorizador TF-IDF del modelo no conoce."""
    tfidf = model.named_steps["preprocessor"].named_transformers_["text"]
    analizar, vocab = tfidf.build_analyzer(), tfidf.vocabulary_
    tokens = [t for texto in textos for t in analizar(str(texto)) if " " not in t]
    return float(np.mean([t not in vocab for t in tokens])) if tokens else 0.0


def psi(ref: pd.Series, nuevo: pd.Series, clases) -> float:
    """Population Stability Index entre dos distribuciones de clases."""
    p = ref.value_counts(normalize=True).reindex(clases, fill_value=0).clip(lower=1e-4)
    q = nuevo.value_counts(normalize=True).reindex(clases, fill_value=0).clip(lower=1e-4)
    return float(((q - p) * np.log(q / p)).sum())


def perfil(model, X, clases, columna_texto="texto_completo") -> dict:
    proba = model.predict_proba(X)
    return {
        "n": len(X),
        "oov": tasa_oov(model, X[columna_texto]),
        "confianza_media": float(proba.max(axis=1).mean()),
        "pct_baja_confianza": float((proba.max(axis=1) < 0.6).mean()),
        "predicciones": pd.Series([clases[i] for i in proba.argmax(axis=1)]),
    }


def comparar(ref: dict, nuevo: dict, clases) -> dict:
    indicadores = {
        "oov_referencia": round(ref["oov"], 4),
        "oov_lote": round(nuevo["oov"], 4),
        "confianza_referencia": round(ref["confianza_media"], 4),
        "confianza_lote": round(nuevo["confianza_media"], 4),
        "pct_baja_confianza_lote": round(nuevo["pct_baja_confianza"], 4),
        "psi_familias": round(psi(ref["predicciones"], nuevo["predicciones"], clases), 4),
    }
    alertas = []
    if nuevo["oov"] > UMBRALES["oov"]:
        alertas.append(f"Palabras desconocidas {nuevo['oov']:.1%} > {UMBRALES['oov']:.0%}")
    caida = ref["confianza_media"] - nuevo["confianza_media"]
    if caida > UMBRALES["caida_confianza"]:
        alertas.append(f"La confianza media cayo {caida:.2f} (> {UMBRALES['caida_confianza']})")
    if indicadores["psi_familias"] > UMBRALES["psi"]:
        alertas.append(f"PSI de familias {indicadores['psi_familias']:.2f} > {UMBRALES['psi']}")
    return {"indicadores": indicadores, "alertas": alertas,
            "recomendacion": "Revisar y reentrenar" if alertas else "Sin deriva relevante"}


def main():
    config = load_config()
    reports = Path(config.get("artifacts", {}).get("reports_dir", "reports"))
    bundle = load_champion(config)
    model, clases = bundle["pipeline"], bundle["classes"]

    df = load_maestro(config)
    df[TARGET] = df[TARGET].map({c: i for i, c in enumerate(clases)})
    *_, df_test, _, _, _ = make_splits(df, TARGET, config)
    ref = perfil(model, df_test, clases)

    X_ext, _ = load_external(config)
    resultado = {
        "umbrales": UMBRALES,
        "referencia": "test reservado (500 materiales no vistos)",
        "lote_nuevo": "items de Ariba en texto libre (249)",
        **comparar(ref, perfil(model, X_ext, clases), clases),
    }
    for k, v in resultado["indicadores"].items():
        print(f"  {k:<26} {v}")
    print("Alertas:", resultado["alertas"] or "ninguna")
    print("Recomendacion:", resultado["recomendacion"])

    path = reports / "monitoreo_drift.json"
    path.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado: {path}")


if __name__ == "__main__":
    main()
