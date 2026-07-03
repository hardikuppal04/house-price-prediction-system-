"""Dataset ingestion and the holdout split.

Both the download and split logic live here so they can be unit-tested and
imported by scripts. Raw data is cached to parquet and is gitignored — it is
never committed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from house_price.config import Config
from house_price.utils import get_logger, load_dataframe, save_dataframe

logger = get_logger(__name__)

# A representative subset of the original Ames 79-column schema. If OpenML
# serves a differently-encoded version under the same id (e.g. one-hot or
# integer-coded columns), these names will be absent and the whole
# feature-engineering plan would silently break. We fail loudly instead.
EXPECTED_AMES_COLUMNS: tuple[str, ...] = (
    "MSSubClass",
    "MSZoning",
    "LotFrontage",
    "LotArea",
    "Neighborhood",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "GrLivArea",
    "FullBath",
    "HalfBath",
    "BsmtFullBath",
    "BsmtHalfBath",
    "GarageCars",
    "GarageArea",
    "PoolQC",
    "SalePrice",
)


class SchemaError(RuntimeError):
    """Raised when a fetched dataset does not match the expected schema."""


def verify_ames_schema(df: pd.DataFrame) -> None:
    """Assert the raw 79-column Ames schema is present.

    Args:
        df: The freshly fetched dataframe (features + target).

    Raises:
        SchemaError: If expected original column names are missing.
    """
    missing = [c for c in EXPECTED_AMES_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            "Fetched dataset does not match the expected raw Ames schema. "
            f"Missing columns: {missing}. "
            f"Got {df.shape[1]} columns, first few: {list(df.columns[:10])}. "
            "OpenML may be serving a differently-encoded version under this id."
        )
    logger.info(
        "Schema check OK: %d rows x %d columns, all expected Ames names present.",
        df.shape[0],
        df.shape[1],
    )


def download_raw(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Fetch the raw dataset from OpenML and cache it as parquet.

    Args:
        cfg: Project configuration.
        force: Re-download even if a cached copy exists.

    Returns:
        The raw dataframe including the target column.
    """
    raw_path = cfg.paths.data_raw / cfg.dataset.raw_filename
    if raw_path.exists() and not force:
        logger.info("Using cached raw data at %s", raw_path)
        df = load_dataframe(raw_path)
        if cfg.dataset.name == "ames":
            verify_ames_schema(df)
        return df

    logger.info(
        "Fetching OpenML dataset id=%s (pins version %s) ...",
        cfg.dataset.openml_id,
        cfg.dataset.openml_version,
    )
    # data_id already pins the exact dataset version; passing `version` too is
    # rejected by fetch_openml, so we keep openml_version in config only as
    # human-readable documentation of which version this id resolves to.
    bunch = fetch_openml(
        data_id=cfg.dataset.openml_id,
        as_frame=True,
        parser="auto",
    )
    df = bunch.frame.copy()

    # OpenML sometimes names the target column differently from the feature
    # schema; normalise it to the configured target name.
    target_name = cfg.dataset.target
    if target_name not in df.columns and bunch.target is not None:
        df[target_name] = np.asarray(bunch.target)

    if cfg.dataset.name == "ames":
        verify_ames_schema(df)

    save_dataframe(df, raw_path)
    logger.info("Cached raw data to %s (%d rows x %d cols)", raw_path, *df.shape)
    return df


def make_split(cfg: Config, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the fixed 80/20 holdout split, stratified on target deciles.

    The split is deterministic given the seed. Stratifying on binned target
    keeps the right-skewed price distribution represented in both partitions.

    Args:
        cfg: Project configuration.
        df: The full raw dataframe including the target.

    Returns:
        ``(train_df, holdout_df)``.
    """
    target = cfg.dataset.target
    strata = pd.qcut(df[target], q=cfg.split.stratify_bins, labels=False, duplicates="drop")

    train_df, holdout_df = train_test_split(
        df,
        test_size=cfg.split.test_size,
        random_state=cfg.seed,
        stratify=strata,
    )
    logger.info(
        "Split: train=%d rows, holdout=%d rows (test_size=%.2f, seed=%d)",
        len(train_df),
        len(holdout_df),
        cfg.split.test_size,
        cfg.seed,
    )
    return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True)


def save_split(cfg: Config, train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> None:
    """Persist the train and holdout partitions to processed/ as parquet."""
    save_dataframe(train_df, cfg.paths.data_processed / cfg.split.train_filename)
    save_dataframe(holdout_df, cfg.paths.data_processed / cfg.split.holdout_filename)


def load_split(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the persisted train and holdout partitions."""
    train_df = load_dataframe(cfg.paths.data_processed / cfg.split.train_filename)
    holdout_df = load_dataframe(cfg.paths.data_processed / cfg.split.holdout_filename)
    return train_df, holdout_df
