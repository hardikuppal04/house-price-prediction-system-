"""CLI: run the tuning comparison (Grid / Randomized / Optuna) and persist results.

Usage:
    python scripts/tune.py [--trials 25]

Writes reports/experiments/tuning_comparison.csv and
reports/experiments/best_configs.json (consumed by scripts/finalize.py).
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from house_price.config import load_config
from house_price.training import (
    load_training_data,
    setup_mlflow,
    tune_boosters_optuna,
    tune_forest_random,
    tune_linear_grid,
)
from house_price.utils import get_logger, set_seed

logger = get_logger("scripts.tune")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tuning comparison.")
    parser.add_argument(
        "--trials",
        type=int,
        default=25,
        help="Optuna trials per booster (and RandomizedSearch iters).",
    )
    args = parser.parse_args()

    cfg = load_config()
    cfg.paths.ensure()
    set_seed(cfg.seed)
    setup_mlflow(cfg)

    X, y = load_training_data(cfg)
    results = []
    results += tune_linear_grid(cfg, X, y)
    results.append(tune_forest_random(cfg, X, y, n_iter=args.trials))
    results += tune_boosters_optuna(cfg, X, y, n_trials=args.trials)

    out_dir = cfg.paths.reports / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([r.as_row() for r in results]).sort_values("cv_log_rmse")
    table.to_csv(out_dir / "tuning_comparison.csv", index=False)

    best_configs = {
        r.model: {"method": r.method, "cv_log_rmse": r.best_score, "params": r.best_params}
        for r in results
    }
    (out_dir / "best_configs.json").write_text(
        json.dumps(best_configs, indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        "Tuning comparison:\n%s",
        table[["model", "method", "cv_log_rmse", "n_evals", "wall_time_s"]]
        .round(5)
        .to_string(index=False),
    )


if __name__ == "__main__":
    main()
