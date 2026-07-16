"""Tests for the FeatureEngineer transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from house_price.feature_engineering import FeatureEngineer


@pytest.fixture
def tiny_frame() -> pd.DataFrame:
    """One hand-computed row so every formula is checked against arithmetic."""
    return pd.DataFrame(
        {
            "YrSold": [2008],
            "YearBuilt": [2000],
            "YearRemodAdd": [2005],
            "GrLivArea": [1500],
            "TotalBsmtSF": [800],
            "FullBath": [2],
            "HalfBath": [1],
            "BsmtFullBath": [1],
            "BsmtHalfBath": [1],
            "OverallQual": [7],
            "OpenPorchSF": [40],
            "EnclosedPorch": [0],
            "3SsnPorch": [10],
            "ScreenPorch": [0],
            "WoodDeckSF": [100],
            "PoolArea": [0],
            "GarageArea": [400],
            "Fireplaces": [1],
        }
    )


def test_engineered_values_hand_checked(tiny_frame: pd.DataFrame) -> None:
    out = FeatureEngineer().fit_transform(tiny_frame)
    row = out.iloc[0]
    assert row["HouseAge"] == 8  # 2008 - 2000
    assert row["RemodAge"] == 3  # 2008 - 2005
    assert row["TotalSF"] == 2300  # 1500 + 800
    assert row["TotalBaths"] == 4.0  # 2 + 0.5 + 1 + 0.5
    assert row["QualByArea"] == 10500  # 7 * 1500
    assert row["PorchSF"] == 150  # 40+0+10+0+100
    assert row["HasPool"] == 0
    assert row["HasGarage"] == 1
    assert row["HasFireplace"] == 1


def test_age_clipped_at_zero(tiny_frame: pd.DataFrame) -> None:
    tiny_frame.loc[0, "YearBuilt"] = 2009  # sold "before" built (data quirk)
    out = FeatureEngineer().fit_transform(tiny_frame)
    assert out.loc[0, "HouseAge"] == 0


def test_input_not_mutated(synthetic_ames: pd.DataFrame) -> None:
    before = synthetic_ames.copy()
    FeatureEngineer().fit_transform(synthetic_ames)
    pd.testing.assert_frame_equal(synthetic_ames, before)


def test_feature_names_out(synthetic_ames: pd.DataFrame) -> None:
    fe = FeatureEngineer().fit(synthetic_ames)
    names = fe.get_feature_names_out()
    assert set(FeatureEngineer.ENGINEERED) <= set(names)
    assert list(names[: synthetic_ames.shape[1]]) == list(synthetic_ames.columns)


def test_missing_column_raises(tiny_frame: pd.DataFrame) -> None:
    bad = tiny_frame.drop(columns=["GrLivArea"])
    with pytest.raises(ValueError, match="GrLivArea"):
        FeatureEngineer().fit(bad)


def test_requires_fit_before_transform(tiny_frame: pd.DataFrame) -> None:
    with pytest.raises(NotFittedError):
        FeatureEngineer().transform(tiny_frame)


def test_sklearn_clone_compatible() -> None:
    clone(FeatureEngineer())  # raises if the estimator contract is violated


def test_nan_bsmt_and_garage_tolerated(tiny_frame: pd.DataFrame) -> None:
    tiny_frame.loc[0, "TotalBsmtSF"] = np.nan
    tiny_frame.loc[0, "GarageArea"] = np.nan
    out = FeatureEngineer().fit_transform(tiny_frame)
    assert out.loc[0, "TotalSF"] == 1500  # NaN basement treated as 0
    assert out.loc[0, "HasGarage"] == 0
