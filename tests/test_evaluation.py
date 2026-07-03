"""Tests for metrics and the consensus-selection rule (pure logic, no CV)."""

from __future__ import annotations

import numpy as np
import pytest

from house_price.evaluation import adjusted_r2, consensus_selection, regression_metrics


def test_perfect_predictions() -> None:
    y = np.log1p(np.array([100_000.0, 200_000.0, 300_000.0]))
    m = regression_metrics(y, y)
    assert m["log_rmse"] == pytest.approx(0.0)
    assert m["dollar_rmse"] == pytest.approx(0.0, abs=1e-6)
    assert m["log_r2"] == pytest.approx(1.0)
    assert m["dollar_mape"] == pytest.approx(0.0, abs=1e-12)


def test_dollar_space_is_expm1_of_log_space() -> None:
    rng = np.random.default_rng(0)
    y_true = np.log1p(rng.uniform(50_000, 500_000, size=50))
    y_pred = y_true + rng.normal(0, 0.1, size=50)
    m = regression_metrics(y_true, y_pred)
    # Hand-computed dollar MAE must match the reported one.
    expected = np.abs(np.expm1(y_true) - np.expm1(y_pred)).mean()
    assert m["dollar_mae"] == pytest.approx(expected)


def test_adjusted_r2_hand_math() -> None:
    # n=30, p=5, r2=0.9 -> 1 - 0.1 * 29/24
    assert adjusted_r2(0.9, 30, 5) == pytest.approx(1 - 0.1 * 29 / 24)
    assert np.isnan(adjusted_r2(0.9, 6, 5))  # n <= p+1 -> undefined


def test_consensus_majority_rule() -> None:
    features = ["a", "b", "c", "d"]
    keep_sets = {
        "m1": {"a", "b"},
        "m2": {"a", "c"},
        "m3": {"a"},
    }
    votes = consensus_selection(keep_sets, features)
    # a: 3 votes, b/c: 1, d: 0; threshold = 1.5
    assert votes.loc["a", "keep"] and votes.loc["a", "votes"] == 3
    assert not votes.loc["b", "keep"]
    assert not votes.loc["d", "keep"]
    assert votes.index[0] == "a"  # sorted by votes


def test_consensus_requires_methods() -> None:
    with pytest.raises(ValueError):
        consensus_selection({}, ["a"])
