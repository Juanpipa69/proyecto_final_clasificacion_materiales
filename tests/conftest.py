"""Fixtures compartidas por las pruebas.

Las pruebas no dependen de models/champion.joblib: entrenan un modelo pequeno y
rapido (LogisticRegression) en una carpeta temporal, con el mismo formato.
"""
from datetime import datetime, timezone

import joblib
import pytest

from src.data.load_data import load_maestro
from src.models.train import TARGET, _get_candidates, build_pipeline
from src.utils.config_loader import load_config


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def df(config):
    return load_maestro(config)


@pytest.fixture(scope="session")
def modelo_path(tmp_path_factory, df, config):
    clases = sorted(df[TARGET].unique())
    y = df[TARGET].map({c: i for i, c in enumerate(clases)})
    pipe = build_pipeline(_get_candidates(42)["LogisticRegression"], config).fit(df, y)
    path = tmp_path_factory.mktemp("modelo") / "modelo_prueba.joblib"
    joblib.dump({
        "pipeline": pipe, "classes": clases, "target": TARGET,
        "algorithm": "LogisticRegression", "feature_set": "completo", "params": {},
        "sklearn_version": "", "trained_at": datetime.now(timezone.utc).isoformat(),
    }, path)
    return path
