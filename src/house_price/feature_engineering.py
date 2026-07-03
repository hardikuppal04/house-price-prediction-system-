"""Custom feature-engineering transformer.

Sklearn-compatible (``BaseEstimator`` + ``TransformerMixin``) so it lives
*inside* the pipeline: engineered features are computed identically at train
and inference time, and leakage is structurally impossible.

Engineered features and their justification:

===========  ==============================================  ==========================================
Feature      Construction                                    Why
===========  ==============================================  ==========================================
HouseAge     YrSold - YearBuilt                              Depreciation proxy; raw years aren't
                                                             linearly meaningful to any model.
RemodAge     YrSold - YearRemodAdd                           Renovation recency drives price beyond age.
TotalSF      GrLivArea + TotalBsmtSF                         Buyers price total usable area, not floors
                                                             separately (collinear cluster seen in EDA).
TotalBaths   Full + 0.5*Half + BsmtFull + 0.5*BsmtHalf       Standard realtor aggregation; a half bath
                                                             is not worth a full one.
QualByArea   OverallQual * GrLivArea                         Price per sq-ft scales with quality —
                                                             multiplicative, invisible to linear models
                                                             unless made explicit.
PorchSF      OpenPorch + Enclosed + 3Ssn + Screen + Deck     Individually sparse/zero-inflated;
                                                             the aggregate captures outdoor amenity.
HasPool      PoolArea > 0                                    Presence effect distinct from size effect
HasGarage    GarageArea > 0                                  (a tiny pool is closer to a big pool than
HasFireplace Fireplaces > 0                                  to no pool at all).
===========  ==============================================  ==========================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Append the engineered Ames features to a DataFrame.

    DataFrame-in / DataFrame-out; input columns are preserved and the
    engineered columns are appended. The transformer is stateless apart from
    remembering input column names for ``get_feature_names_out``.
    """

    ENGINEERED: tuple[str, ...] = (
        "HouseAge",
        "RemodAge",
        "TotalSF",
        "TotalBaths",
        "QualByArea",
        "PorchSF",
        "HasPool",
        "HasGarage",
        "HasFireplace",
    )

    _REQUIRED: tuple[str, ...] = (
        "YrSold",
        "YearBuilt",
        "YearRemodAdd",
        "GrLivArea",
        "TotalBsmtSF",
        "FullBath",
        "HalfBath",
        "BsmtFullBath",
        "BsmtHalfBath",
        "OverallQual",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "WoodDeckSF",
        "PoolArea",
        "GarageArea",
        "Fireplaces",
    )

    def _validate(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame.")
        missing = [c for c in self._REQUIRED if c not in X.columns]
        if missing:
            raise ValueError(f"FeatureEngineer: missing required columns {missing}")

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        self._validate(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        self._validate(X)
        out = X.copy()

        # Ages can go mildly negative on data-entry quirks (sold before
        # "built"); clip at 0 rather than propagating nonsense.
        out["HouseAge"] = (out["YrSold"] - out["YearBuilt"]).clip(lower=0)
        out["RemodAge"] = (out["YrSold"] - out["YearRemodAdd"]).clip(lower=0)

        out["TotalSF"] = out["GrLivArea"] + out["TotalBsmtSF"].fillna(0)
        out["TotalBaths"] = (
            out["FullBath"].fillna(0)
            + 0.5 * out["HalfBath"].fillna(0)
            + out["BsmtFullBath"].fillna(0)
            + 0.5 * out["BsmtHalfBath"].fillna(0)
        )
        out["QualByArea"] = out["OverallQual"] * out["GrLivArea"]
        out["PorchSF"] = (
            out["OpenPorchSF"]
            + out["EnclosedPorch"]
            + out["3SsnPorch"]
            + out["ScreenPorch"]
            + out["WoodDeckSF"]
        )
        out["HasPool"] = (out["PoolArea"] > 0).astype(int)
        out["HasGarage"] = (out["GarageArea"].fillna(0) > 0).astype(int)
        out["HasFireplace"] = (out["Fireplaces"] > 0).astype(int)
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        base = self.feature_names_in_ if input_features is None else np.asarray(input_features)
        return np.concatenate([base, np.asarray(self.ENGINEERED, dtype=object)])
