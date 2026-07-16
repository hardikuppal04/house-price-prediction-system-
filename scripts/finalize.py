"""CLI: fit the final model, save the artifact, and run THE holdout evaluation.

Usage:
    python scripts/finalize.py [--model NAME]

Without --model, the best tuned configuration (lowest CV log-RMSE in
best_configs.json) wins. This script performs the project's single touch of
the holdout set; the learning curve is computed on training data only.
"""

from __future__ import annotations

import argparse
import json

from house_price.config import load_config
from house_price.training import (
    build_final_pipeline,
    evaluate_holdout,
    finalize_model,
    load_training_data,
    make_cv,
    setup_mlflow,
)
from house_price.utils import get_logger, set_seed
from house_price.visualization import plot_learning_curve

logger = get_logger("scripts.finalize")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize model + holdout eval.")
    parser.add_argument(
        "--model", default=None, help="Override the winner (a name from best_configs.json)."
    )
    args = parser.parse_args()

    cfg = load_config()
    cfg.paths.ensure()
    set_seed(cfg.seed)
    setup_mlflow(cfg)

    configs = json.loads(
        (cfg.paths.reports / "experiments" / "best_configs.json").read_text(encoding="utf-8")
    )
    name = args.model or min(configs, key=lambda k: configs[k]["cv_log_rmse"])
    chosen = configs[name]
    logger.info(
        "Final model: %s (CV log-RMSE %.5f, method %s)",
        name,
        chosen["cv_log_rmse"],
        chosen["method"],
    )

    finalize_model(cfg, name, chosen["params"], chosen["cv_log_rmse"])

    # Learning curve on training data only (bias/variance evidence for M6/M8).
    X, y = load_training_data(cfg)
    plot_learning_curve(
        build_final_pipeline(cfg, name, chosen["params"]),
        X,
        y,
        make_cv(cfg),
        save_path=cfg.paths.figures / "learning_curve_final.png",
    )

    metrics = evaluate_holdout(cfg)
    logger.info(
        "HOLDOUT (single touch) — log-RMSE %.5f | R2 %.4f | $MAE %.0f | MAPE %.4f",
        metrics["log_rmse"],
        metrics["log_r2"],
        metrics["dollar_mae"],
        metrics["dollar_mape"],
    )


if __name__ == "__main__":
    main()
