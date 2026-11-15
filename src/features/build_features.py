"""Preprocesador de features para la clasificacion de materiales.

Conjuntos de features disponibles (FEATURE_SETS):
- "completo": TF-IDF sobre texto_completo + OneHot(marca, um) + escalado numerico
- "solo_texto_breve": solo TF-IDF sobre texto_breve

'grupo_sap' se excluye siempre: es la etiqueta SAP actual (ruidosa) y usarla
como feature mezclaria la etiqueta vieja con la que queremos predecir.
Todo el preprocesamiento va dentro del Pipeline, asi que se ajusta solo con
los datos de entrenamiento de cada fold (sin data leakage).
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TEXT_COL = "texto_completo"
CATEGORICAL_COLS = ["marca", "um"]
NUMERIC_COLS = ["precio_usd", "filas_me2m", "items_ariba"]

FEATURE_SETS = {
    "completo": {"text_col": "texto_completo", "include_tabular": True},
    "solo_texto_breve": {"text_col": "texto_breve", "include_tabular": False},
}


def build_preprocessor(
    config: dict | None = None,
    text_col: str = TEXT_COL,
    include_tabular: bool = True,
) -> ColumnTransformer:
    """Devuelve un ColumnTransformer sin ajustar, listo para ir en un Pipeline."""
    tfidf_cfg = (config or {}).get("tfidf", {})
    tfidf = TfidfVectorizer(
        max_features=tfidf_cfg.get("max_features", 5000),
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
        min_df=tfidf_cfg.get("min_df", 2),
        sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
        strip_accents="unicode",
    )
    transformers = [("text", tfidf, text_col)]
    if include_tabular:
        transformers += [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)
