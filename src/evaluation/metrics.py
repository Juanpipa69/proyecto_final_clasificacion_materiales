"""Graficas de evaluacion del modelo campeon guardado.

Carga models/champion.joblib (generado por src.models.train), lo evalua en el
mismo test reservado y guarda en reports/:
  - confusion_matrix_champion.png
  - f1_por_clase_champion.png
  - cv_comparacion_modelos.png
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

from src.data.load_data import load_maestro
from src.models.train import TARGET, make_splits
from src.utils.config_loader import load_config


def load_champion(config) -> dict:
    arts = config.get("artifacts", {})
    path = Path(arts.get("models_dir", "models")) / arts.get("model_file", "champion.joblib")
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Primero ejecuta: python -m src.models.train")
    return joblib.load(path)


def main():
    config = load_config()
    reports = Path(config.get("artifacts", {}).get("reports_dir", "reports"))
    bundle = load_champion(config)
    model, classes = bundle["pipeline"], bundle["classes"]
    name = f"{bundle['algorithm']} ({bundle['feature_set']})"
    print(f"Campeon: {name}")

    df = load_maestro(config)
    df[TARGET] = df[TARGET].map({c: i for i, c in enumerate(classes)})
    _, _, df_test, _, _, y_test = make_splits(df, TARGET, config)
    y_pred = model.predict(df_test)

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=classes, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Matriz de confusion - {name} (test)")
    fig.tight_layout()
    fig.savefig(reports / "confusion_matrix_champion.png", dpi=150)
    plt.close(fig)

    f1s = f1_score(y_test, y_pred, average=None)
    order = np.argsort(f1s)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh([classes[i] for i in order], f1s[order], color="#2a6fdb")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1-score")
    ax.set_title(f"F1 por clase - {name} (test)")
    fig.tight_layout()
    fig.savefig(reports / "f1_por_clase_champion.png", dpi=150)
    plt.close(fig)

    summary_path = reports / "metrics_summary.json"
    if summary_path.exists():
        cv = json.loads(summary_path.read_text(encoding="utf-8")).get("cv_algoritmos", {})
        if cv:
            names = sorted(cv, key=lambda k: cv[k]["f1_macro_cv"])
            means = [cv[k]["f1_macro_cv"] for k in names]
            stds = [cv[k]["std"] for k in names]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            bars = ax.barh(names, means, xerr=stds, color="#2a6fdb", capsize=4)
            ax.bar_label(bars, fmt="%.4f", padding=6)
            ax.set_xlim(min(means) - 0.03, 1.01)
            ax.set_xlabel("F1-macro (validacion cruzada 5 folds, media +/- desv.)")
            ax.set_title("Comparacion de algoritmos")
            fig.tight_layout()
            fig.savefig(reports / "cv_comparacion_modelos.png", dpi=150)
            plt.close(fig)

    print(f"F1-macro test: {f1s.mean():.4f}")
    print(f"Graficas guardadas en {reports}")


if __name__ == "__main__":
    main()
