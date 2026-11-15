"""Inferencia: clasifica materiales nuevos con el modelo campeon guardado.

Uso desde la terminal (texto obligatorio, resto opcional):
    python -m src.models.predict --texto "GUANTE NITRILO TALLA L" --marca ansell --um par

Uso desde codigo:
    from src.models.predict import Clasificador
    clf = Clasificador()
    clf.predecir([{"texto_breve": "GUANTE NITRILO TALLA L"}])
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data.load_data import _normalize_text
from src.utils.config_loader import load_config

DEFAULTS = {"texto_largo": "", "marca": "SIN_MARCA", "um": "",
            "precio_usd": 0.0, "filas_me2m": 0, "items_ariba": 0}


class Clasificador:
    """Carga el modelo una sola vez y clasifica uno o varios materiales."""

    def __init__(self, model_path: str | Path | None = None):
        if model_path is None:
            arts = load_config().get("artifacts", {})
            model_path = Path(arts.get("models_dir", "models")) / arts.get(
                "model_file", "champion.joblib")
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No existe {model_path}. Primero ejecuta: python -m src.models.train")
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.classes = bundle["classes"]
        self.info = {k: bundle[k] for k in ("algorithm", "feature_set", "trained_at")}

    @staticmethod
    def _preparar(items: list[dict]) -> pd.DataFrame:
        """Aplica la misma normalizacion que src.data.load_data al entrenar."""
        rows = []
        for item in items:
            if not str(item.get("texto_breve", "")).strip():
                raise ValueError("Cada material necesita 'texto_breve'.")
            r = {**DEFAULTS, **item}
            breve, largo = str(r["texto_breve"]), str(r["texto_largo"] or "")
            rows.append({
                "texto_breve": breve,
                "texto_completo": " | ".join(
                    p for p in (_normalize_text(breve), _normalize_text(largo)) if p),
                "marca": _normalize_text(r["marca"] or "SIN_MARCA"),
                "um": _normalize_text(r["um"] or ""),
                "precio_usd": float(r["precio_usd"] or 0),
                "filas_me2m": int(r["filas_me2m"] or 0),
                "items_ariba": int(r["items_ariba"] or 0),
            })
        return pd.DataFrame(rows)

    def predecir(self, items: list[dict], top_k: int = 3) -> list[dict]:
        X = self._preparar(items)
        proba = self.pipeline.predict_proba(X)
        out = []
        for p in proba:
            order = p.argsort()[::-1][:top_k]
            out.append({
                "familia": self.classes[order[0]],
                "confianza": round(float(p[order[0]]), 4),
                "alternativas": [
                    {"familia": self.classes[i], "probabilidad": round(float(p[i]), 4)}
                    for i in order[1:]
                ],
            })
        return out


def main():
    parser = argparse.ArgumentParser(description="Clasifica un material por familia.")
    parser.add_argument("--texto", required=True, help="Texto breve del material")
    parser.add_argument("--largo", default="", help="Texto largo (opcional)")
    parser.add_argument("--marca", default="SIN_MARCA")
    parser.add_argument("--um", default="")
    parser.add_argument("--precio", type=float, default=0.0)
    args = parser.parse_args()

    clf = Clasificador()
    item = {"texto_breve": args.texto, "texto_largo": args.largo,
            "marca": args.marca, "um": args.um, "precio_usd": args.precio}
    print(json.dumps({"modelo": clf.info, "prediccion": clf.predecir([item])[0]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
