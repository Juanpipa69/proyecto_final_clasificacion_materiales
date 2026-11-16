"""Pipeline completo y reproducible en un solo comando.

Pasos: entrenar (con validacion de datos, CV, tuning y evaluacion) ->
graficas y lineas base (solo familia) -> registro en MLflow.

Uso:
    python -m src.pipelines.run_pipeline                    # familia + registro @champion
    python -m src.pipelines.run_pipeline --target categoria # experimento adicional
    python -m src.pipelines.run_pipeline --no-mlflow        # sin MLflow
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation import baselines, metrics
from src.models.tracking import log_run
from src.models.train import TARGET, TARGETS, run_training
from src.utils.config_loader import load_config


def run(target: str = TARGET, use_mlflow: bool = True) -> dict:
    config = load_config()
    reports = Path(config.get("artifacts", {}).get("reports_dir", "reports"))

    print("=" * 60 + f"\nPASO 1/3 - Entrenamiento ({target})\n" + "=" * 60)
    result = run_training(config, target=target)

    artifacts = [Path("configs/config.yaml"), result["summary_path"], result["test_report_path"]]
    if target == TARGET:
        print("\n" + "=" * 60 + "\nPASO 2/3 - Graficas y lineas base\n" + "=" * 60)
        metrics.main()
        baselines.main()
        artifacts += [reports / f for f in (
            "confusion_matrix_champion.png", "f1_por_clase_champion.png",
            "cv_comparacion_modelos.png", "baselines.md", "baselines.json")]
    else:
        print("\nPASO 2/3 - Graficas y lineas base: solo aplican al objetivo principal (familia)")

    print("\n" + "=" * 60 + "\nPASO 3/3 - Registro en MLflow\n" + "=" * 60)
    if use_mlflow:
        # Solo el modelo de familia (objetivo principal) se registra como @champion
        log_run(config, result, artifacts, register=(target == TARGET))
    else:
        print("  Omitido (--no-mlflow)")
    print("\nPipeline terminado.")
    return result


def main():
    parser = argparse.ArgumentParser(description="Pipeline de clasificacion de materiales.")
    parser.add_argument("--target", choices=list(TARGETS), default="familia")
    parser.add_argument("--no-mlflow", action="store_true", help="No registrar en MLflow")
    args = parser.parse_args()
    run(TARGETS[args.target], use_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
