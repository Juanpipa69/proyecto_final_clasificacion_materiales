"""Funciones de evaluacion reutilizables y conjunto de prueba externo.

- classification_metrics: accuracy, precision/recall/F1 macro, ROC-AUC y PR-AUC
  multiclase (one-vs-rest, promedio macro).
- load_external: 249 items de eventos Ariba redactados en texto libre (hoja
  Verdad_items_texto_libre), con su familia real. Sirve como prueba externa:
  el modelo nunca los ve al entrenar y su redaccion es distinta a la del maestro.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.data.load_data import _normalize_text


def classification_metrics(pipe, X, y, n_classes: int) -> dict:
    """Metricas multiclase. `y` debe venir codificada como 0..n_classes-1."""
    y_pred = pipe.predict(X)
    out = {
        "accuracy": accuracy_score(y, y_pred),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro"),
        "f1_weighted": f1_score(y, y_pred, average="weighted"),
    }
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        labels = np.arange(n_classes)
        out["roc_auc_ovr_macro"] = roc_auc_score(
            y, proba, multi_class="ovr", average="macro", labels=labels
        )
        out["pr_auc_macro"] = average_precision_score(
            label_binarize(y, classes=labels), proba, average="macro"
        )
    return {k: round(float(v), 4) for k, v in out.items()}


def load_external(config) -> tuple[pd.DataFrame, pd.Series]:
    """Items de texto libre de Ariba con las mismas columnas que espera el modelo.

    Estos items no traen marca, UM, precio ni historico: se llenan con valores
    neutros. Por eso miden sobre todo la capacidad de generalizar desde el texto.
    """
    path = Path(config["data"]["raw_dir"]) / config["data"]["files"]["maestro"]
    sheet = config["data"]["sheets"].get("externo", "Verdad_items_texto_libre")
    raw = pd.read_excel(path, sheet_name=sheet)
    desc = raw["Descripci\u00f3n del \u00edtem"].astype(str)
    X = pd.DataFrame({
        "texto_completo": desc.map(_normalize_text),
        "texto_breve": desc,
        "marca": "sin_marca",
        "um": "",
        "precio_usd": 0.0,
        "filas_me2m": 0,
        "items_ariba": 0,
    })
    return X, raw["Familia real"]
