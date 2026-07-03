"""CLI: create and persist the fixed 80/20 holdout split.

Usage:
    python scripts/make_split.py

The split is deterministic given the configured seed and is saved to
data/processed/. The holdout partition is touched exactly once, at final
evaluation (Milestone 5).
"""

from __future__ import annotations

from house_price.config import load_config
from house_price.data import download_raw, make_split, save_split
from house_price.utils import get_logger, set_seed

logger = get_logger("scripts.make_split")


def main() -> None:
    cfg = load_config()
    cfg.paths.ensure()
    set_seed(cfg.seed)

    df = download_raw(cfg)
    train_df, holdout_df = make_split(cfg, df)
    save_split(cfg, train_df, holdout_df)
    logger.info(
        "Saved split to %s (train) and %s (holdout).",
        cfg.paths.data_processed / cfg.split.train_filename,
        cfg.paths.data_processed / cfg.split.holdout_filename,
    )


if __name__ == "__main__":
    main()
