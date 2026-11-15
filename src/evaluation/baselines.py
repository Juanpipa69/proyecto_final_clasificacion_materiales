"""Lineas base para contextualizar el desempeno del modelo campeon.

Entrena modelos simples con el mismo conjunto de desarrollo que el campeon
(train + val) y los compara en el mismo test reservado. Ademas evalua el
campeon en un test "limpio" (sin texto_breve identico a alguno de desarrollo).
Guarda reports/baselines.json y reports/baselines.md.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import load_maestro
from src.evaluation.metrics import load_champion
from src.models.train import TARGET, make_splits
from src.utils.config_loader import load_config

warnings.filterwarnings("ignore")
NUM = ["precio_usd", "filas_me2m", "items_ariba"]


def _f1(model, X, y):
    return round(f1_score(y, model.predict(X), average="macro"), 4)


def main():
    config = load_config()
    seed = config.get("split", {}).get("random_state", 42)
    reports = Path(config.get("artifacts", {}).get("reports_dir", "reports"))
    bundle = load_champion(config)
    champion, classes = bundle["pipeline"], bundle["classes"]

    df = load_maestro(config)
    df[TARGET] = df[TARGET].map({c: i for i, c in enumerate(classes)})
    df_train, df_val, df_test, _, _, y_test = make_splits(df, TARGET, config)
    df_dev = pd.concat([df_train, df_val])
    y_dev = df_dev[TARGET]

    baselines = {
        "Azar (clase mas frecuente)": Pipeline([
            ("p", ColumnTransformer([("n", "passthrough", ["precio_usd"])])),
            ("c", DummyClassifier(strategy="most_frequent")),
        ]),
        "Agrupacion SAP actual (grupo_sap)": Pipeline([
            ("p", ColumnTransformer([("g", OneHotEncoder(handle_unknown="ignore"), ["grupo_sap"])])),
            ("c", LogisticRegression(max_iter=1000)),
        ]),
        "Sin texto (precio, marca, UM)": Pipeline([
            ("p", ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["marca", "um"]),
                ("num", StandardScaler(), NUM),
            ])),
            ("c", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),
        ]),
        "Solo texto_breve (TF-IDF + LogReg)": Pipeline([
            ("p", ColumnTransformer([("t", TfidfVectorizer(), "texto_breve")])),
            ("c", LogisticRegression(max_iter=1000)),
        ]),
    }

    results = {}
    for name, pipe in baselines.items():
        pipe.fit(df_dev, y_dev)
        results[name] = _f1(pipe, df_test, y_test)
        print(f"  {name:<45} F1-macro test = {results[name]:.4f}")

    champ_name = f"Modelo del proyecto ({bundle['algorithm']})"
    results[champ_name] = _f1(champion, df_test, y_test)
    print(f"  {champ_name:<45} F1-macro test = {results[champ_name]:.4f}")

    mask = ~df_test["texto_breve"].isin(set(df_dev["texto_breve"]))
    n_dup = int((~mask).sum())
    clean_f1 = _f1(champion, df_test[mask], y_test[mask])
    print(f"\n  Test sin texto_breve repetido ({int(mask.sum())} de {len(df_test)} filas, "
          f"{n_dup} quitadas): F1-macro = {clean_f1:.4f}")

    out = {
        "f1_macro_test": results,
        "test_sin_duplicados": {
            "filas": int(mask.sum()), "duplicados_quitados": n_dup, "f1_macro": clean_f1,
        },
    }
    (reports / "baselines.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = ["# Lineas base - F1-macro en test", "", "| Enfoque | F1-macro |", "|---|---|"]
    lines += [f"| {k} | {v:.4f} |" for k, v in results.items()]
    lines += ["", f"Test sin texto_breve repetido: {int(mask.sum())} filas "
              f"({n_dup} quitadas), F1-macro = {clean_f1:.4f}", ""]
    (reports / "baselines.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nGuardado: {reports / 'baselines.json'} y {reports / 'baselines.md'}")


if __name__ == "__main__":
    main()
