"""Tests for schema verification and the deterministic holdout split."""

from __future__ import annotations

import pandas as pd
import pytest

from house_price.config import load_config
from house_price.data import SchemaError, make_split, verify_ames_schema


def test_verify_schema_accepts_valid_frame(synthetic_ames: pd.DataFrame) -> None:
    # Should not raise: the synthetic frame carries the expected Ames names.
    verify_ames_schema(synthetic_ames)


def test_verify_schema_rejects_encoded_frame(synthetic_ames: pd.DataFrame) -> None:
    # Simulate OpenML serving a differently-encoded version (renamed columns).
    encoded = synthetic_ames.rename(columns={"OverallQual": "feature_17"})
    with pytest.raises(SchemaError):
        verify_ames_schema(encoded)


def test_split_is_deterministic(synthetic_ames: pd.DataFrame) -> None:
    cfg = load_config()
    train_a, holdout_a = make_split(cfg, synthetic_ames)
    train_b, holdout_b = make_split(cfg, synthetic_ames)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(holdout_a, holdout_b)


def test_split_sizes_and_no_overlap(synthetic_ames: pd.DataFrame) -> None:
    cfg = load_config()
    train_df, holdout_df = make_split(cfg, synthetic_ames)
    n = len(synthetic_ames)
    assert len(train_df) + len(holdout_df) == n
    # 20% holdout, allow rounding slack from stratification.
    assert abs(len(holdout_df) - round(0.2 * n)) <= 1
    overlap = pd.merge(train_df, holdout_df, how="inner")
    assert overlap.empty
