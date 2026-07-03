"""Shared pytest fixtures.

The synthetic frame mimics the parts of the Ames schema used across tests so
unit tests never depend on a network fetch from OpenML.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_ames(rng: np.random.Generator) -> pd.DataFrame:
    """A small Ames-like frame with the columns downstream code relies on.

    Includes informative-missingness (PoolQC), a true numeric gap
    (LotFrontage), ordinal quality scales, and a high-cardinality categorical
    (Neighborhood).
    """
    n = 120
    year_built = rng.integers(1900, 2010, size=n)
    df = pd.DataFrame(
        {
            "Id": np.arange(1, n + 1),
            "MSSubClass": rng.choice([20, 30, 60, 70], size=n),
            "MSZoning": rng.choice(["RL", "RM", "FV"], size=n),
            "LotFrontage": rng.normal(70, 20, size=n),
            "LotArea": rng.integers(3000, 15000, size=n),
            "Neighborhood": rng.choice(
                ["CollgCr", "Veenker", "Crawfor", "NridgHt", "OldTown"], size=n
            ),
            "OverallQual": rng.integers(1, 11, size=n),
            "OverallCond": rng.integers(1, 11, size=n),
            "YearBuilt": year_built,
            "YearRemodAdd": year_built + rng.integers(0, 20, size=n),
            "YrSold": rng.integers(2006, 2011, size=n),
            "GrLivArea": rng.integers(500, 4000, size=n),
            "TotalBsmtSF": rng.integers(0, 2000, size=n),
            "Fireplaces": rng.integers(0, 3, size=n),
            "FullBath": rng.integers(0, 4, size=n),
            "HalfBath": rng.integers(0, 3, size=n),
            "BsmtFullBath": rng.integers(0, 3, size=n),
            "BsmtHalfBath": rng.integers(0, 2, size=n),
            "GarageCars": rng.integers(0, 4, size=n),
            "GarageArea": rng.integers(0, 900, size=n),
            "ExterQual": rng.choice(["Po", "Fa", "TA", "Gd", "Ex"], size=n),
            "KitchenQual": rng.choice(["Po", "Fa", "TA", "Gd", "Ex"], size=n),
            "OpenPorchSF": rng.integers(0, 300, size=n),
            "EnclosedPorch": rng.integers(0, 200, size=n),
            "ScreenPorch": rng.integers(0, 200, size=n),
            "3SsnPorch": rng.integers(0, 100, size=n),
            "WoodDeckSF": rng.integers(0, 400, size=n),
        }
    )

    # Informative missingness: most houses have no pool -> PoolQC is NaN and
    # PoolArea is 0, mirroring the real data's coupling.
    pool_qc = np.array(["NA"] * n, dtype=object)
    has_pool = rng.random(n) < 0.05
    pool_qc[has_pool] = rng.choice(["Fa", "Gd", "Ex"], size=has_pool.sum())
    pool_qc[pool_qc == "NA"] = np.nan
    df["PoolQC"] = pool_qc
    df["PoolArea"] = np.where(has_pool, rng.integers(100, 800, size=n), 0)

    # GarageYrBlt is informative-NaN where there is no garage.
    no_garage = df["GarageCars"] == 0
    df.loc[no_garage, "GarageArea"] = 0
    garage_yr = year_built + rng.integers(0, 5, size=n)
    df["GarageYrBlt"] = np.where(no_garage, np.nan, garage_yr)

    # True gap: sprinkle a few genuine NaNs into LotFrontage.
    gap_idx = rng.choice(n, size=8, replace=False)
    df.loc[gap_idx, "LotFrontage"] = np.nan

    # Target with a right-skewed shape driven mostly by quality and area.
    base = 30000 + df["OverallQual"] * 15000 + df["GrLivArea"] * 40
    noise = rng.normal(0, 15000, size=n)
    df["SalePrice"] = np.clip(base + noise, 40000, None).round().astype(int)
    return df
