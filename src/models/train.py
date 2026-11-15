"""Entrenamiento, seleccion, optimizacion y evaluacion del modelo campeon.

Flujo:
 1. Carga y validacion de datos.
 2. Split por grupos (material + duplicados juntos): desarrollo (80%) y test (20%).
 3. Validacion cruzada estratificada por grupos (5 folds) de 5 algoritmos.
 4. Comparacion de conjuntos de features para el mejor algoritmo.
 5. Optimizacion de hiperparametros (RandomizedSearchCV) y comparacion antes/despues.
 6. Evaluacion final en test (nunca usado antes) y en el set externo de texto libre.
 7. Guardado del modelo en models/champion.joblib y metricas en reports/.

El test solo se usa una vez, al final. Todo el preprocesamiento va dentro del
Pipeline, por lo que en cada fold se ajusta solo con los datos de entrenamiento.
"""
from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data.load_data import load_maestro
from src.data.validate import validate_dataset
from src.evaluation.evaluate import classification_metrics, load_external
from src.features.build_features import FEATURE_SETS, build_preprocessor
from src.utils.config_loader import load_config

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

TARGET = "familia_cod"

# Espacios de busqueda de hiperparametros (prefijo "clf__" = el clasificador)
PARAM_SPACES = {
    "LogisticRegression": {"clf__C": [0.1, 0.3, 1, 3, 10, 30, 100]},
    "LinearSVC": {"clf__estimator__C": [0.03, 0.1, 0.3, 1, 3, 10]},
    "DecisionTree": {"clf__max_depth": [None, 20, 40, 80], "clf__min_samples_leaf": [1, 2, 5]},
    "RandomForest": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 40, 80],
        "clf__min_samples_leaf": [1, 2],
        "clf__max_features": ["sqrt", 0.05, 0.1],
    },
    "XGBoost": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [4, 6, 8],
        "clf__learning_rate": [0.05, 0.1, 0.2],
    },
}
TFIDF_SPACE = {
    "preprocessor__text__ngram_range": [(1, 1), (1, 2)],
    "preprocessor__text__min_df": [1, 2],
}


def _get_candidates(random_state: int = 42) -> dict:
    """Algoritmos candidatos con hiperparametros por defecto razonables."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced", random_state=random_state
        ),
        # LinearSVC no da probabilidades: se calibra para poder medir ROC/PR-AUC
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", random_state=random_state), cv=3
        ),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=random_state),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric="mlogloss",
            random_state=random_state, n_jobs=-1, verbosity=0,
        ),
    }


def build_pipeline(estimator, config, feature_set: str = "completo") -> Pipeline:
    pre = build_preprocessor(config, **FEATURE_SETS[feature_set])
    return Pipeline([("preprocessor", pre), ("clf", estimator)])


def _group_fold(df, y, groups, n_splits, random_state):
    """Devuelve (idx_resto, idx_fold) del primer fold estratificado por grupos."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return next(sgkf.split(df, y, groups))


def make_splits(df, target_col, config):
    """Split train/val/test estratificado y sin compartir grupos entre conjuntos.

    Si existe `grupo_id`, un material y sus duplicados quedan siempre en el
    mismo conjunto. Proporciones aproximadas: test_size y val_size del config.
    """
    split_cfg = config.get("split", {})
    test_size = split_cfg.get("test_size", 0.20)
    val_size = split_cfg.get("val_size", 0.10)
    random_state = split_cfg.get("random_state", 42)

    df = df.reset_index(drop=True)
    groups = df["grupo_id"] if "grupo_id" in df.columns else df.index.to_series()
    y = df[target_col]

    idx_tv, idx_test = _group_fold(df, y, groups, round(1 / test_size), random_state)
    df_tv, df_test = df.iloc[idx_tv].reset_index(drop=True), df.iloc[idx_test]

    n_val = round((1 - test_size) / val_size)
    idx_train, idx_val = _group_fold(
        df_tv, df_tv[target_col], groups.iloc[idx_tv].reset_index(drop=True), n_val, random_state
    )
    df_train, df_val = df_tv.iloc[idx_train], df_tv.iloc[idx_val]

    if "grupo_id" in df.columns:
        g_tr, g_va, g_te = (set(d["grupo_id"]) for d in (df_train, df_val, df_test))
        assert not (g_tr & g_te or g_tr & g_va or g_va & g_te), "Grupos compartidos entre splits"

    print(f"  Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}"
          f"  (sin grupos compartidos)")
    return (df_train, df_val, df_test,
            df_train[target_col], df_val[target_col], df_test[target_col])


def _cv_score(pipe, X, y, groups, cv) -> tuple[float, float]:
    scores = cross_val_score(pipe, X, y, groups=groups, cv=cv, scoring="f1_macro", n_jobs=1)
    return float(scores.mean()), float(scores.std())


def run_training(config=None):
    config = config or load_config()
    seed = config.get("split", {}).get("random_state", 42)
    reports = Path(config.get("artifacts", {}).get("reports_dir", "reports"))
    models_dir = Path(config.get("artifacts", {}).get("models_dir", "models"))
    reports.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    # 1. Datos
    print("Cargando y validando datos...")
    df = load_maestro(config)
    report = validate_dataset(df, config)
    print(report.summary())
    if not report.passed:
        raise ValueError("La validacion de datos fallo; revisa los errores arriba.")

    encoder = LabelEncoder()
    df[TARGET] = encoder.fit_transform(df[TARGET])
    class_names = list(encoder.classes_)
    n_classes = len(class_names)

    # 2. Split: desarrollo (train+val) para CV y test intocable
    print("\nGenerando splits...")
    df_train, df_val, df_test, _, _, y_test = make_splits(df, TARGET, config)
    df_dev = pd.concat([df_train, df_val]).reset_index(drop=True)
    y_dev, groups_dev = df_dev[TARGET], df_dev["grupo_id"]
    cv = StratifiedGroupKFold(
        n_splits=config.get("cv", {}).get("n_splits", 5), shuffle=True, random_state=seed
    )
    print(f"  Desarrollo para CV: {len(df_dev)} | Test reservado: {len(df_test)}")

    # 3. Comparacion de algoritmos con validacion cruzada
    print(f"\n[1/4] Validacion cruzada ({cv.n_splits} folds por grupos)...")
    cv_results = {}
    for name, est in _get_candidates(seed).items():
        t0 = time.time()
        mean, std = _cv_score(build_pipeline(est, config), df_dev, y_dev, groups_dev, cv)
        cv_results[name] = {"f1_macro_cv": round(mean, 4), "std": round(std, 4),
                            "segundos": round(time.time() - t0, 1)}
        print(f"  {name:<20} F1-macro CV = {mean:.4f} +/- {std:.4f}  ({time.time() - t0:.0f}s)")
    best_algo = max(cv_results, key=lambda k: cv_results[k]["f1_macro_cv"])
    print(f"  -> Mejor algoritmo: {best_algo}")

    # 4. Comparacion de conjuntos de features para el mejor algoritmo
    print(f"\n[2/4] Conjuntos de features para {best_algo}...")
    fs_results = {}
    for fs in FEATURE_SETS:
        est = _get_candidates(seed)[best_algo]
        mean, std = _cv_score(build_pipeline(est, config, fs), df_dev, y_dev, groups_dev, cv)
        fs_results[fs] = {"f1_macro_cv": round(mean, 4), "std": round(std, 4)}
        print(f"  {fs:<20} F1-macro CV = {mean:.4f} +/- {std:.4f}")
    best_fs = max(fs_results, key=lambda k: fs_results[k]["f1_macro_cv"])
    print(f"  -> Mejor conjunto de features: {best_fs}")

    # 5. Optimizacion de hiperparametros
    print(f"\n[3/4] Optimizando hiperparametros de {best_algo} ({best_fs})...")
    space = {**PARAM_SPACES[best_algo], **TFIDF_SPACE}
    n_total = 1
    for v in space.values():
        n_total *= len(v)
    search = RandomizedSearchCV(
        build_pipeline(_get_candidates(seed)[best_algo], config, best_fs),
        param_distributions=space,
        n_iter=min(config.get("tuning", {}).get("n_iter", 12), n_total),
        scoring="f1_macro", cv=cv, random_state=seed, n_jobs=1, refit=True,
        return_train_score=True,
    )
    search.fit(df_dev, y_dev, groups=groups_dev)
    before = fs_results[best_fs]["f1_macro_cv"]
    after = round(float(search.best_score_), 4)
    i = search.best_index_
    train_cv = round(float(search.cv_results_["mean_train_score"][i]), 4)
    best_params = {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in search.best_params_.items()}
    print(f"  Antes: {before:.4f} | Despues: {after:.4f}")
    print(f"  F1 train (promedio folds) = {train_cv:.4f} vs CV = {after:.4f}"
          f"  (brecha {train_cv - after:.4f})")
    print(f"  Mejores parametros: {best_params}")
    model = search.best_estimator_  # reentrenado con todo el conjunto de desarrollo

    # 6. Evaluacion final: test reservado y set externo
    print("\n[4/4] Evaluacion final...")
    test_metrics = classification_metrics(model, df_test, y_test, n_classes)
    report_test = classification_report(y_test, model.predict(df_test), target_names=class_names)
    X_ext, y_ext_names = load_external(config)
    known = y_ext_names.isin(class_names)
    y_ext = encoder.transform(y_ext_names[known])
    ext_metrics = classification_metrics(model, X_ext[known], y_ext, n_classes)
    ext_metrics["n_items"] = int(known.sum())
    print(f"  TEST     : {test_metrics}")
    print(f"  EXTERNO  : {ext_metrics}")
    print(f"\nReporte test:\n\n{report_test}")

    # 7. Artefactos
    model_path = models_dir / config.get("artifacts", {}).get("model_file", "champion.joblib")
    joblib.dump({
        "pipeline": model,
        "classes": class_names,
        "algorithm": best_algo,
        "feature_set": best_fs,
        "params": best_params,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, model_path)

    (reports / "test_report_champion.txt").write_text(report_test, encoding="utf-8")
    metrics = {
        "champion": best_algo,
        "feature_set": best_fs,
        "best_params": best_params,
        "f1_macro_cv_antes_tuning": before,
        "f1_macro_cv_despues_tuning": after,
        "f1_macro_train_cv": train_cv,
        "f1_macro_test": test_metrics["f1_macro"],
        "test": test_metrics,
        "externo_texto_libre": ext_metrics,
        "cv_algoritmos": cv_results,
        "cv_feature_sets": fs_results,
        "classes": class_names,
        "split": {"desarrollo": len(df_dev), "test": len(df_test), "agrupado_por": "grupo_id",
                  "cv_folds": cv.n_splits},
        "modelo": str(model_path),
    }
    (reports / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nModelo guardado en {model_path}")
    print(f"Metricas guardadas en {reports / 'metrics_summary.json'}")
    print(f"\nCAMPEON: {best_algo} ({best_fs}) | F1-macro test = {test_metrics['f1_macro']:.4f}"
          f" | externo = {ext_metrics['f1_macro']:.4f}")
    return metrics


if __name__ == "__main__":
    run_training()
