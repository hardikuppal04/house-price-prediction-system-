"""Tests for preprocessing transformers and the assembled pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from house_price.preprocessing import (
    ColumnDropper,
    InformativeNAFiller,
    NeighborhoodGroupedImputer,
    QualityOrdinalMapper,
    SkewedLog1pTransformer,
    TypeCaster,
    build_preprocessor,
    drop_partial_sale_outliers,
)

# ---------------------------------------------------------------------------
# Individual transformers
# ---------------------------------------------------------------------------


def test_informative_na_filler(synthetic_ames: pd.DataFrame) -> None:
    out = InformativeNAFiller().fit_transform(synthetic_ames)
    assert out["PoolQC"].isna().sum() == 0
    assert (out.loc[synthetic_ames["PoolQC"].isna(), "PoolQC"] == "None").all()
    assert out["GarageYrBlt"].isna().sum() == 0
    # True gaps must be left alone for the dedicated imputer.
    assert out["LotFrontage"].isna().sum() == synthetic_ames["LotFrontage"].isna().sum()


def test_grouped_imputer_uses_fit_medians_only(synthetic_ames: pd.DataFrame) -> None:
    imp = NeighborhoodGroupedImputer().fit(synthetic_ames)
    out = imp.transform(synthetic_ames)
    assert out["LotFrontage"].isna().sum() == 0

    # A missing value is filled with its neighborhood's *fit-time* median.
    na_idx = synthetic_ames.index[synthetic_ames["LotFrontage"].isna()][0]
    hood = synthetic_ames.loc[na_idx, "Neighborhood"]
    assert out.loc[na_idx, "LotFrontage"] == pytest.approx(imp.group_medians_[hood])

    # Unseen neighborhood at inference -> global median, not an error/NaN.
    new = synthetic_ames.iloc[[na_idx]].copy()
    new["Neighborhood"] = "Nowhere"
    filled = imp.transform(new)
    assert filled["LotFrontage"].iloc[0] == pytest.approx(imp.global_median_)


def test_quality_ordinal_mapper_order() -> None:
    df = pd.DataFrame({"ExterQual": ["None", "Po", "Fa", "TA", "Gd", "Ex", "??"]})
    out = QualityOrdinalMapper().fit_transform(df)
    assert out["ExterQual"].tolist() == [0, 1, 2, 3, 4, 5, -1]


def test_type_caster_msubclass(synthetic_ames: pd.DataFrame) -> None:
    out = TypeCaster().fit_transform(synthetic_ames)
    assert out["MSSubClass"].dtype == object or str(out["MSSubClass"].dtype) == "str"


def test_column_dropper(synthetic_ames: pd.DataFrame) -> None:
    out = ColumnDropper(("Id", "NotThere")).fit_transform(synthetic_ames)
    assert "Id" not in out.columns
    names = ColumnDropper(("Id",)).fit(synthetic_ames).get_feature_names_out()
    assert "Id" not in names


def test_skewed_log1p_learned_at_fit(synthetic_ames: pd.DataFrame) -> None:
    tr = SkewedLog1pTransformer(threshold=0.5).fit(synthetic_ames.select_dtypes(np.number))
    # Column list is a fit-time artifact; binary flags excluded.
    assert all(synthetic_ames[c].nunique() > 2 for c in tr.columns_)
    out = tr.transform(synthetic_ames.select_dtypes(np.number))
    for col in tr.columns_:
        assert out[col].max() <= np.log1p(synthetic_ames[col].max()) + 1e-9


def test_drop_partial_sale_outliers() -> None:
    df = pd.DataFrame({"GrLivArea": [1500, 4500, 4600], "SalePrice": [200000, 150000, 500000]})
    out = drop_partial_sale_outliers(df)
    # Only the big-but-cheap house goes; the big-and-expensive one stays.
    assert len(out) == 2
    assert 150000 not in out["SalePrice"].values


# ---------------------------------------------------------------------------
# Assembled pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoding", ["ohe", "ordinal", "target"])
def test_pipeline_output_is_clean(synthetic_ames: pd.DataFrame, encoding: str) -> None:
    X = synthetic_ames.drop(columns=["SalePrice"])
    y = np.log1p(synthetic_ames["SalePrice"])
    pre = build_preprocessor(encoding=encoding, seed=0)
    out = pre.fit_transform(X, y)

    assert isinstance(out, pd.DataFrame)
    assert out.isna().sum().sum() == 0
    assert np.isfinite(out.to_numpy(dtype=float)).all()
    assert "Id" not in out.columns
    # Engineered features survive into the encoded output.
    assert "TotalSF" in out.columns and "QualByArea" in out.columns
    # Transform on unseen-like data yields the same schema.
    again = pre.transform(X.head(5))
    assert list(again.columns) == list(out.columns)


def test_pipeline_encoding_variants_differ(synthetic_ames: pd.DataFrame) -> None:
    X = synthetic_ames.drop(columns=["SalePrice"])
    y = np.log1p(synthetic_ames["SalePrice"])
    ohe_cols = build_preprocessor("ohe").fit_transform(X, y).columns
    ord_cols = build_preprocessor("ordinal").fit_transform(X, y).columns
    tgt_cols = build_preprocessor("target", seed=0).fit_transform(X, y).columns
    # Ordinal collapses quality dummies into single integer columns.
    assert "ExterQual" in ord_cols and "ExterQual" not in ohe_cols
    assert len(ord_cols) < len(ohe_cols)
    # Target encoding collapses Neighborhood dummies into one numeric column.
    assert "Neighborhood" in tgt_cols
    assert not any(str(c).startswith("Neighborhood_") for c in tgt_cols)


def test_pipeline_is_cloneable_and_refittable(synthetic_ames: pd.DataFrame) -> None:
    X = synthetic_ames.drop(columns=["SalePrice"])
    y = np.log1p(synthetic_ames["SalePrice"])
    pre = build_preprocessor("target", seed=0)
    cloned = clone(pre)  # required for use inside GridSearch/cross_val_score
    cloned.fit(X, y)
    assert cloned.transform(X).shape[0] == len(X)
