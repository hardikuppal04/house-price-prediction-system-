"""CLI: run the M4 experiment stages, logging to MLflow and reports/experiments.

Usage:
    python scripts/run_experiments.py --stage encoding
    python scripts/run_experiments.py --stage zoo --encoding target
    python scripts/run_experiments.py --stage selection --encoding target
"""

from __future__ import annotations

import argparse

from house_price.config import load_config
from house_price.training import (
    run_encoding_experiment,
    run_feature_selection,
    run_model_zoo,
    setup_mlflow,
)
from house_price.utils import get_logger, set_seed

logger = get_logger("scripts.run_experiments")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M4 experiments.")
    parser.add_argument("--stage", choices=["encoding", "zoo", "selection"], required=True)
    parser.add_argument(
        "--encoding",
        default="ohe",
        choices=["ohe", "ordinal", "target"],
        help="Encoding for zoo/selection stages (winner of the " "encoding stage).",
    )
    args = parser.parse_args()

    cfg = load_config()
    cfg.paths.ensure()
    set_seed(cfg.seed)
    setup_mlflow(cfg)

    out_dir = cfg.paths.reports / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "encoding":
        table = run_encoding_experiment(cfg)
        path = out_dir / "encoding_comparison.csv"
    elif args.stage == "zoo":
        table = run_model_zoo(cfg, args.encoding)
        path = out_dir / f"model_zoo_{args.encoding}.csv"
    else:
        votes, table = run_feature_selection(cfg, args.encoding)
        votes.to_csv(out_dir / "feature_selection_votes.csv")
        path = out_dir / "feature_selection_comparison.csv"

    table.to_csv(path)
    cols = [
        c
        for c in (
            "log_rmse_mean",
            "log_rmse_std",
            "dollar_mae_mean",
            "dollar_mape_mean",
            "fit_time_s",
        )
        if c in table.columns
    ]
    logger.info("Results written to %s\n%s", path, table[cols].round(5).to_string())


if __name__ == "__main__":
    main()
