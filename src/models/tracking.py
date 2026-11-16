"""Registro de experimentos y modelos en MLflow.

Por cada entrenamiento crea un run padre con parametros, metricas, artefactos
(config, reportes, graficas) y el modelo con firma (signature) y ejemplo de
entrada. Cada algoritmo evaluado en validacion cruzada queda como run hijo,
para compararlos en la interfaz de MLflow.

Si register=True, el modelo se registra en el Model Registry y recibe el
alias definido en config (por defecto "champion"), de modo que se puede cargar
con: mlflow.pyfunc.load_model("models:/<nombre>@champion").

Si MLflow no esta instalado o falla (por ejemplo, por politicas del equipo
como AppLocker), se avisa y el pipeline sigue: el modelo ya quedo en models/.
"""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "desconocido"
    except Exception:
        return "desconocido"


def _short(param: str) -> str:
    """'clf__estimator__C' -> 'C'; 'preprocessor__text__min_df' -> 'tfidf_min_df'."""
    if param.startswith("preprocessor__text__"):
        return "tfidf_" + param.split("__")[-1]
    return param.split("__")[-1]


def _serialization_kwargs(model) -> dict:
    """MLflow 3 guarda modelos sklearn con skops (formato seguro, no ejecuta codigo al
    cargar) y exige declarar los tipos no estandar. Se declaran solo los tipos del
    modelo que nosotros mismos entrenamos. Si skops no lo soporta, se usa cloudpickle."""
    try:
        from skops.io import dumps, get_untrusted_types
        return {"skops_trusted_types": get_untrusted_types(data=dumps(model)),
                "pip_requirements": _requirements(include_skops=True)}
    except Exception:
        return {"serialization_format": "cloudpickle", "pip_requirements": _requirements()}


def _requirements(include_skops: bool = False) -> list[str]:
    """Dependencias minimas y fijadas para cargar el modelo (evita copiar todo uv.lock)."""
    import numpy
    import pandas
    import sklearn
    reqs = [f"scikit-learn=={sklearn.__version__}", f"pandas=={pandas.__version__}",
            f"numpy=={numpy.__version__}"]
    if include_skops:
        import skops
        reqs.append(f"skops=={skops.__version__}")
    return reqs


def log_run(config: dict, result: dict, artifacts: list[Path], register: bool = False):
    """Registra un entrenamiento en MLflow. Devuelve el run_id o None si no se pudo."""
    try:
        import mlflow
        import sklearn
        import xgboost
        from mlflow.models import infer_signature
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        print(f"  [MLflow] No disponible ({exc}); se omite el registro.")
        return None

    cfg = config.get("mlflow", {})
    m = result["metrics"]
    model, X_ex = result["model"], result["X_example"]
    target = m["target"]
    experiment = cfg.get("experiment_name", "clasificacion-materiales-familia")
    if target != "familia_cod":
        nivel = target.replace("_cod", "")
        experiment = (experiment.replace("familia", nivel) if "familia" in experiment
                      else f"{experiment}-{nivel}")

    try:
        mlflow.set_tracking_uri(cfg.get("tracking_uri", "sqlite:///mlflow.db"))
        mlflow.set_experiment(experiment)

        with mlflow.start_run(run_name=f"{m['champion']}-{m['feature_set']}") as run:
            mlflow.set_tags({
                "target": target,
                "algoritmo": m["champion"],
                "feature_set": m["feature_set"],
                "git_commit": _git_commit(),
                "datos": "ficticios (semilla 42)",
            })
            mlflow.log_params({
                "target": target,
                "algoritmo": m["champion"],
                "feature_set": m["feature_set"],
                "cv_folds": m["split"]["cv_folds"],
                "split_agrupado_por": m["split"]["agrupado_por"],
                "n_desarrollo": m["split"]["desarrollo"],
                "n_test": m["split"]["test"],
                "n_clases": len(m["classes"]),
                "python": platform.python_version(),
                "sklearn": sklearn.__version__,
                "xgboost": xgboost.__version__,
                **{_short(k): v for k, v in m["best_params"].items()},
            })
            metrics = {
                "f1_macro_cv_antes_tuning": m["f1_macro_cv_antes_tuning"],
                "f1_macro_cv_despues_tuning": m["f1_macro_cv_despues_tuning"],
                "f1_macro_train_cv": m["f1_macro_train_cv"],
                **{f"test_{k}": v for k, v in m["test"].items()},
                **{f"externo_{k}": v for k, v in m.get("externo_texto_libre", {}).items()
                   if k != "n_items"},
            }
            mlflow.log_metrics(metrics)

            for algo, r in m["cv_algoritmos"].items():
                with mlflow.start_run(run_name=f"cv-{algo}", nested=True):
                    mlflow.set_tags({"target": target, "tipo": "comparacion_cv"})
                    mlflow.log_param("algoritmo", algo)
                    mlflow.log_metrics({"f1_macro_cv": r["f1_macro_cv"], "f1_macro_cv_std": r["std"],
                                        "segundos": r["segundos"]})

            for path in artifacts:
                if Path(path).exists():
                    mlflow.log_artifact(str(path))

            signature = infer_signature(X_ex, model.predict_proba(X_ex))
            reg_name = cfg.get("registered_model_name", "clasificacion-materiales")
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                input_example=X_ex.head(2),
                registered_model_name=reg_name if register else None,
                **_serialization_kwargs(model),
            )

            if register and info.registered_model_version:
                alias = cfg.get("champion_alias", "champion")
                client = MlflowClient()
                version = info.registered_model_version
                client.set_registered_model_alias(reg_name, alias, version)
                # La prediccion de MLflow es el indice de clase: se guarda el mapeo
                client.set_model_version_tag(reg_name, version, "clases", ",".join(m["classes"]))
                client.set_model_version_tag(reg_name, version, "f1_macro_test",
                                             str(m["test"]["f1_macro"]))
                print(f"  [MLflow] Modelo registrado: {reg_name} v{info.registered_model_version}"
                      f" con alias @{alias}")

            print(f"  [MLflow] Run {run.info.run_id[:8]} en experimento '{experiment}'")
            return run.info.run_id
    except Exception as exc:  # el pipeline no debe caerse por el tracking
        print(f"  [MLflow] Error al registrar ({type(exc).__name__}: {exc}); se continua sin MLflow.")
        return None
