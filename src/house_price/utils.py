"""Cross-cutting helpers: logging, seeding, and small IO utilities."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring the root handler once.

    Using the stdlib ``logging`` module (not ``print``) keeps output
    timestamped, levelled, and redirectable in the API and scripts.
    """
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    return logger


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and the ``PYTHONHASHSEED`` env var for reproducibility.

    Model-specific seeds (sklearn ``random_state``, Optuna samplers) are passed
    explicitly at call sites; this covers the global sources of nondeterminism.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_dataframe(path: Path) -> pd.DataFrame:
    """Read a parquet DataFrame, with a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {path}. "
            "Run scripts/download_data.py and scripts/make_split.py first."
        )
    return pd.read_parquet(path)
