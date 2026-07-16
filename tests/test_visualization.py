"""Tests for EDA compute functions and figure generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CI/test runs

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from house_price.visualization import (
    missingness_table,
    numeric_skewness,
    plot_missingness,
    plot_neighborhood_prices,
    plot_target_distribution,
    target_correlations,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_missingness_classifies_informative_vs_gap(synthetic_ames: pd.DataFrame) -> None:
    table = missingness_table(synthetic_ames)
    assert table.loc["PoolQC", "kind"] == "informative"
    assert table.loc["LotFrontage", "kind"] == "true gap"
    # Only columns that actually have missing values appear.
    assert "GrLivArea" not in table.index
    # Percentages are consistent with counts.
    expected_pct = round(table.loc["LotFrontage", "n_missing"] / len(synthetic_ames) * 100, 2)
    assert table.loc["LotFrontage", "pct_missing"] == pytest.approx(expected_pct)


def test_missingness_sorted_descending(synthetic_ames: pd.DataFrame) -> None:
    table = missingness_table(synthetic_ames)
    assert table["pct_missing"].is_monotonic_decreasing


def test_numeric_skewness_filters_and_sorts(synthetic_ames: pd.DataFrame) -> None:
    skew = numeric_skewness(synthetic_ames, threshold=0.1)
    assert (skew.abs() > 0.1).all()
    assert skew.abs().is_monotonic_decreasing


def test_target_correlations_excludes_target(synthetic_ames: pd.DataFrame) -> None:
    corr = target_correlations(synthetic_ames, "SalePrice")
    assert "SalePrice" not in corr.index
    # SalePrice is constructed from OverallQual and GrLivArea in the fixture,
    # so both must rank among the strongest correlates.
    assert {"OverallQual", "GrLivArea"} <= set(corr.head(3).index)


def test_plots_return_figures_and_save(synthetic_ames: pd.DataFrame, tmp_path: Path) -> None:
    table = missingness_table(synthetic_ames)
    out = tmp_path / "figs" / "missing.png"
    fig = plot_missingness(table, save_path=out)
    assert fig is not None
    assert out.exists() and out.stat().st_size > 0

    fig2 = plot_target_distribution(synthetic_ames["SalePrice"], save_path=tmp_path / "t.png")
    assert (tmp_path / "t.png").exists()
    assert fig2.get_axes()[0].get_title() != ""

    fig3 = plot_neighborhood_prices(synthetic_ames, "SalePrice", save_path=tmp_path / "n.png")
    assert (tmp_path / "n.png").exists()
    assert fig3 is not None
