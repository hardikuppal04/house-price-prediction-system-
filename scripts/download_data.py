"""CLI: download and cache the raw dataset from OpenML.

Usage:
    python scripts/download_data.py [--force]
"""

from __future__ import annotations

import argparse

from house_price.config import load_config
from house_price.data import download_raw
from house_price.utils import get_logger, set_seed

logger = get_logger("scripts.download_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw data from OpenML.")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached.")
    args = parser.parse_args()

    cfg = load_config()
    cfg.paths.ensure()
    set_seed(cfg.seed)

    df = download_raw(cfg, force=args.force)
    logger.info("Raw data ready: %d rows x %d columns.", *df.shape)


if __name__ == "__main__":
    main()
