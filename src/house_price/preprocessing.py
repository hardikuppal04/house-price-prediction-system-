"""Preprocessing transformers and the pipeline builder.

Everything here composes into a single sklearn ``Pipeline`` — imputation,
type fixes, feature engineering, encoding, scaling — so nothing is ever fit
outside a training fold and leakage is impossible by construction.

Encoding variants (compared head-to-head in Milestone 4):

- ``"ohe"``      — every categorical one-hot encoded (baseline).
- ``"ordinal"``  — genuine quality scales get an explicit
                   None<Po<Fa<TA<Gd<Ex ordinal map; other categoricals OHE.
- ``"target"``   — Neighborhood via sklearn's cross-fitted ``TargetEncoder``
                   (CV-safe), quality scales ordinal, the rest OHE.
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.utils.validation import check_is_fitted

from house_price.data import INFORMATIVE_NA_COLUMNS
from house_price.feature_engineering import FeatureEngineer
from house_price.utils import get_logger

logger = get_logger(__name__)

# Explicit quality scale — shared by every quality-graded column. "None"
# (feature absent, injected by InformativeNAFiller) sits below "Po" (poor).
QUALITY_SCALE: tuple[str, ...] = ("None", "Po", "Fa", "TA", "Gd", "Ex")

# Columns genuinely graded on that scale. Deliberately excludes look-alike
# ordinals with different level sets (BsmtExposure, GarageFinish, ...).
QUALITY_COLUMNS: tuple[str, ...] = (
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
    "PoolQC",
)

Encoding = Literal["ohe", "ordinal", "target", "raw"]


class TypeCaster(BaseEstimator, TransformerMixin):
    """Cast integer-coded categoricals to string.

    MSSubClass values like 20/60/120 are building-type codes, not quantities;
    left numeric they would be scaled and fed to linear models as magnitudes.
    """

    CAST_TO_STR: tuple[str, ...] = ("MSSubClass",)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TypeCaster":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        out = X.copy()
        for col in self.CAST_TO_STR:
            if col in out.columns:
                out[col] = out[col].astype(str)
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_ if input_features is None else np.asarray(input_features)


class InformativeNAFiller(BaseEstimator, TransformerMixin):
    """Make informative missingness explicit instead of imputing it away.

    Per the Ames data dictionary, NaN in these columns means "feature absent"
    (no pool, no garage, no basement). Categorical columns get an explicit
    ``"None"`` level; the numeric companions get 0:

    - ``GarageYrBlt`` -> 0 when there is no garage.
    - ``MasVnrArea`` -> 0 (NaN co-occurs with MasVnrType NaN = no veneer;
      the EDA's naive "true gap" label for this column is overridden here,
      deliberately).
    """

    NUMERIC_ZERO: tuple[str, ...] = ("GarageYrBlt", "MasVnrArea")

    def __init__(self, columns: tuple[str, ...] = INFORMATIVE_NA_COLUMNS) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "InformativeNAFiller":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        out = X.copy()
        for col in self.columns:
            if col in out.columns and col not in self.NUMERIC_ZERO:
                out[col] = out[col].fillna("None")
        for col in self.NUMERIC_ZERO:
            if col in out.columns:
                out[col] = out[col].fillna(0)
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_ if input_features is None else np.asarray(input_features)


class NeighborhoodGroupedImputer(BaseEstimator, TransformerMixin):
    """Impute a numeric column with per-group medians learned at fit time.

    Used for LotFrontage: street frontage is strongly neighborhood-dependent
    (EDA), so the neighborhood median beats the global one. Unseen groups at
    inference fall back to the global median. Medians are learned only in
    ``fit`` — inside a CV fold, that means from the training fold alone.
    """

    def __init__(self, column: str = "LotFrontage", group: str = "Neighborhood") -> None:
        self.column = column
        self.group = group

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NeighborhoodGroupedImputer":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.group_medians_ = X.groupby(self.group, observed=True)[self.column].median()
        self.global_median_ = float(X[self.column].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "group_medians_")
        out = X.copy()
        fill = out[self.group].map(self.group_medians_).fillna(self.global_median_)
        out[self.column] = out[self.column].fillna(fill)
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "group_medians_")
        return self.feature_names_in_ if input_features is None else np.asarray(input_features)


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop columns that are identifiers, not features (tolerant if absent)."""

    def __init__(self, columns: tuple[str, ...] = ("Id",)) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ColumnDropper":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        return X.drop(columns=[c for c in self.columns if c in X.columns])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        base = self.feature_names_in_ if input_features is None else np.asarray(input_features)
        return np.asarray([c for c in base if c not in self.columns], dtype=object)


class SkewedLog1pTransformer(BaseEstimator, TransformerMixin):
    """log1p-transform numeric columns whose training-fold skew exceeds a threshold.

    Column selection happens in ``fit`` (training fold only). Non-negative
    columns only; binary/near-constant columns are excluded because skew is
    meaningless there. Helps linear models; monotone no-op for trees.
    """

    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SkewedLog1pTransformer":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        numeric = X.select_dtypes(include=np.number)
        skew = numeric.skew()
        self.columns_ = [
            col
            for col in numeric.columns
            if abs(skew[col]) > self.threshold
            and numeric[col].min() >= 0
            and numeric[col].nunique() > 2
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "columns_")
        out = X.copy()
        for col in self.columns_:
            out[col] = np.log1p(out[col])
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "columns_")
        return self.feature_names_in_ if input_features is None else np.asarray(input_features)


class CategoricalNAFiller(BaseEstimator, TransformerMixin):
    """Fill remaining non-numeric NaNs with an explicit "Missing" level.

    Used only in the ``raw`` (CatBoost-native) variant: CatBoost rejects NaN
    in categorical features, and true-gap categoricals (e.g. one Electrical
    row) survive InformativeNAFiller by design.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CategoricalNAFiller":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        out = X.copy()
        for col in out.select_dtypes(exclude=np.number).columns:
            out[col] = out[col].fillna("Missing")
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_ if input_features is None else np.asarray(input_features)


class QualityOrdinalMapper(BaseEstimator, TransformerMixin):
    """Map quality-scale columns to integers via the explicit QUALITY_SCALE.

    Preferred over ``OrdinalEncoder`` here because it works on whichever
    subset of quality columns is present (the ColumnTransformer selects them
    dynamically) and the None<Po<Fa<TA<Gd<Ex order is stated once, in code,
    rather than inferred from data. Unknown levels map to -1.
    """

    _MAP = {level: i for i, level in enumerate(QUALITY_SCALE)}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "QualityOrdinalMapper":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            out[col] = X[col].map(self._MAP).fillna(-1).astype(int)
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_ if input_features is None else np.asarray(input_features)


# ---------------------------------------------------------------------------
# Column selectors (module-level so the fitted pipeline stays picklable)
# ---------------------------------------------------------------------------


def _select_quality(X: pd.DataFrame) -> list[str]:
    return [c for c in QUALITY_COLUMNS if c in X.columns]


def _select_nominal(X: pd.DataFrame, exclude: tuple[str, ...]) -> list[str]:
    non_numeric = X.select_dtypes(exclude=np.number).columns
    return [c for c in non_numeric if c not in exclude]


def drop_partial_sale_outliers(df: pd.DataFrame, target: str = "SalePrice") -> pd.DataFrame:
    """Remove the documented Ames partial-sale outliers (train-time only).

    Two houses >4000 sq ft sold under $300k — flagged by the dataset author
    as partial sales, i.e. recorded prices that do not reflect market value.
    This is a *training data curation* decision applied before fitting, never
    to evaluation data. Adoption is decided by CV evidence (M3/M4), not taste.
    """
    mask = (df["GrLivArea"] > 4000) & (df[target] < 300000)
    if mask.any():
        logger.info("Dropping %d partial-sale outlier row(s).", int(mask.sum()))
    return df.loc[~mask].reset_index(drop=True)


def build_preprocessor(
    encoding: Encoding = "ohe",
    *,
    seed: int = 42,
    log1p_skewed: bool = False,
) -> Pipeline:
    """Assemble the full preprocessing pipeline (imputation -> FE -> encoding).

    Args:
        encoding: One of ``"ohe"``, ``"ordinal"``, ``"target"`` (see module
            docstring).
        seed: Random state for the cross-fitted TargetEncoder.
        log1p_skewed: Also log1p-transform skewed numeric features (helps
            linear models; irrelevant to trees).

    Returns:
        An unfitted sklearn Pipeline producing a dense, fully numeric,
        NaN-free DataFrame (pandas output is enabled for name traceability).
    """
    if encoding not in ("ohe", "ordinal", "target", "raw"):
        raise ValueError(f"Unknown encoding {encoding!r}")

    cleaning_steps: list = [
        ("types", TypeCaster()),
        ("informative_na", InformativeNAFiller()),
        ("lot_frontage", NeighborhoodGroupedImputer()),
        ("features", FeatureEngineer()),
        ("drop_id", ColumnDropper(("Id",))),
    ]

    if encoding == "raw":
        # CatBoost-native variant: cleaning + FE only, categoricals kept as
        # strings for the model's own categorical handling. No scaling — the
        # only consumer is a tree booster.
        cleaning_steps.append(("cat_na", CategoricalNAFiller()))
        pipeline = Pipeline(cleaning_steps)
        pipeline.set_output(transform="pandas")
        return pipeline

    numeric_pipe = Pipeline(
        [
            # Safety-net imputer: all *known* gaps are handled upstream, but a
            # NaN reaching a model should never be possible by construction.
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    transformers: list = [
        ("numeric", numeric_pipe, make_column_selector(dtype_include=np.number)),
    ]
    nominal_excluded: tuple[str, ...] = ()
    if encoding in ("ordinal", "target"):
        transformers.append(("quality", QualityOrdinalMapper(), _select_quality))
        nominal_excluded += QUALITY_COLUMNS
    if encoding == "target":
        # Explicit CV generator: seeds the encoder's internal cross-fitting
        # (the thing that makes it leakage-safe) with our project seed.
        encoder_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        transformers.append(
            (
                "neighborhood",
                TargetEncoder(target_type="continuous", cv=encoder_cv),
                ["Neighborhood"],
            ),
        )
        nominal_excluded += ("Neighborhood",)
    transformers.append(("nominal", ohe, partial(_select_nominal, exclude=nominal_excluded)))

    steps = cleaning_steps
    if log1p_skewed:
        steps.append(("log1p_skewed", SkewedLog1pTransformer()))
    steps.append(
        (
            "encode",
            ColumnTransformer(
                transformers=transformers, remainder="drop", verbose_feature_names_out=False
            ),
        )
    )

    pipeline = Pipeline(steps)
    pipeline.set_output(transform="pandas")
    return pipeline
