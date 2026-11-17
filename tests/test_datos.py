"""Pruebas de carga, validacion, split sin fuga y preprocesamiento."""
import numpy as np
import pandas as pd

from src.data.load_data import _normalize_text
from src.data.validate import validate_dataset
from src.features.build_features import FEATURE_SETS, build_preprocessor
from src.models.train import TARGET, make_splits


def test_carga_tiene_columnas_y_grupos(df):
    assert len(df) == 2500
    for col in ["texto_completo", "marca", "um", "precio_usd", "grupo_id", TARGET]:
        assert col in df.columns
    # 100 duplicados declarados -> 2400 grupos
    assert df["grupo_id"].nunique() == 2400


def test_normalizacion_de_texto():
    assert _normalize_text("  Válvula   DE Ñandú ") == "valvula de nandu"
    assert _normalize_text(None) == ""


def test_validacion_pasa_con_datos_correctos(df, config):
    assert validate_dataset(df, config).passed


def test_validacion_falla_si_falta_el_objetivo(df, config):
    assert not validate_dataset(df.drop(columns=[TARGET]), config).passed


def test_split_sin_grupos_compartidos(df, config):
    tr, va, te, *_ = make_splits(df, TARGET, config)
    assert len(tr) + len(va) + len(te) == len(df)
    g_tr, g_va, g_te = set(tr["grupo_id"]), set(va["grupo_id"]), set(te["grupo_id"])
    assert not (g_tr & g_te) and not (g_tr & g_va) and not (g_va & g_te)


def test_split_estratificado_todas_las_clases_en_test(df, config):
    *_, y_te = make_splits(df, TARGET, config)
    assert set(y_te.unique()) == set(df[TARGET].unique())


def test_split_reproducible(df, config):
    a = make_splits(df, TARGET, config)[2]
    b = make_splits(df, TARGET, config)[2]
    assert list(a.index) == list(b.index)


def test_preprocesador_no_usa_la_etiqueta_ruidosa(df, config):
    for nombre, kwargs in FEATURE_SETS.items():
        pre = build_preprocessor(config, **kwargs)
        columnas = [c for _, _, cols in pre.transformers for c in np.atleast_1d(cols)]
        assert "grupo_sap" not in columnas, nombre
        assert TARGET not in columnas, nombre


def test_preprocesador_transforma(df, config):
    X = build_preprocessor(config).fit_transform(df.head(200))
    assert X.shape[0] == 200 and X.shape[1] > 10
    assert not pd.isna(X.sum())
