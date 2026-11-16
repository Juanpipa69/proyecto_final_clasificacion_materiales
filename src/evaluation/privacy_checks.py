"""Verificaciones de privacidad y seguridad del modelo campeon (Fase 6).

1. Datos personales: busca patrones de datos personales (correos, telefonos
   moviles, numeros de 8-10 digitos tipo cedula) en las columnas que usa el
   modelo y en el vocabulario TF-IDF guardado dentro del artefacto.
2. Membership inference: mide si un atacante que solo ve las probabilidades
   puede distinguir materiales de entrenamiento vs. no vistos (AUC ~0.5 = no).
3. Equidad entre clases: brecha entre la mejor y la peor familia en test.

Guarda reports/privacidad_seguridad.json.
Uso: python -m src.evaluation.privacy_checks
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.data.load_data import load_maestro
from src.evaluation.metrics import load_champion
from src.models.train import FEATURE_COLS, TARGET, make_splits
from src.utils.config_loader import load_config

PATRONES = {
    "correo": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "telefono_movil_co": re.compile(r"\b3\d{9}\b"),
    "numero_8_10_digitos": re.compile(r"\b\d{8,10}\b"),
}


def _buscar_patrones(textos) -> dict:
    texto = " ".join(map(str, textos))
    return {nombre: len(p.findall(texto)) for nombre, p in PATRONES.items()}


def main():
    config = load_config()
    reports = Path(config.get("artifacts", {}).get("reports_dir", "reports"))
    bundle = load_champion(config)
    model, classes = bundle["pipeline"], bundle["classes"]

    df = load_maestro(config)
    df[TARGET] = df[TARGET].map({c: i for i, c in enumerate(classes)})
    df_train, df_val, df_test, _, _, y_test = make_splits(df, TARGET, config)
    df_dev = pd.concat([df_train, df_val])

    # 1. Datos personales en entradas del modelo y en el artefacto
    entradas = _buscar_patrones(df[["texto_breve", "texto_completo", "marca"]].values.ravel())
    vocab = model.named_steps["preprocessor"].named_transformers_["text"].vocabulary_
    en_vocab = _buscar_patrones(vocab.keys())
    print(f"Columnas usadas por el modelo: {FEATURE_COLS}")
    print(f"Patrones de datos personales en entradas: {entradas}")
    print(f"Vocabulario guardado en el modelo: {len(vocab)} terminos | patrones: {en_vocab}")

    # 2. Membership inference por umbral de confianza
    conf_dev = model.predict_proba(df_dev).max(axis=1)
    conf_test = model.predict_proba(df_test).max(axis=1)
    es_miembro = np.r_[np.ones(len(conf_dev)), np.zeros(len(conf_test))]
    auc_mia = roc_auc_score(es_miembro, np.r_[conf_dev, conf_test])
    print(f"Confianza media: entrenamiento={conf_dev.mean():.4f} | no vistos={conf_test.mean():.4f}")
    print(f"Membership inference (umbral de confianza): AUC={auc_mia:.3f}  (0.5 = no distingue)")

    # 3. Equidad entre clases
    f1_clase = f1_score(y_test, model.predict(df_test), average=None)
    brecha = float(f1_clase.max() - f1_clase.min())
    print(f"F1 por familia en test: min={f1_clase.min():.4f} max={f1_clase.max():.4f}"
          f" brecha={brecha:.4f}")

    out = {
        "columnas_usadas_por_el_modelo": FEATURE_COLS,
        "patrones_datos_personales_en_entradas": entradas,
        "vocabulario_en_artefacto": {"terminos": len(vocab), "patrones": en_vocab},
        "membership_inference": {
            "metodo": "umbral sobre la confianza maxima (atacante de caja negra)",
            "auc": round(float(auc_mia), 4),
            "confianza_media_entrenamiento": round(float(conf_dev.mean()), 4),
            "confianza_media_no_vistos": round(float(conf_test.mean()), 4),
        },
        "equidad_entre_clases": {
            "f1_por_familia": {c: round(float(v), 4) for c, v in zip(classes, f1_clase)},
            "brecha_max_min": round(brecha, 4),
        },
    }
    path = reports / "privacidad_seguridad.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado: {path}")


if __name__ == "__main__":
    main()
