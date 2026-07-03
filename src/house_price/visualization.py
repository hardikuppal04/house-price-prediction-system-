"""EDA computations and plots.

Each analysis has a compute function (returns a DataFrame/Series — unit-testable
without rendering) and a plot function (returns a matplotlib Figure, optionally
saved). Notebooks call these and narrate; they contain no logic themselves.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from house_price.data import INFORMATIVE_NA_COLUMNS
from house_price.utils import get_logger

logger = get_logger(__name__)


def _finalize(fig: Figure, save_path: Path | None) -> Figure:
    """Tighten layout and optionally persist the figure to disk."""
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure to %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------

def missingness_table(
    df: pd.DataFrame,
    informative_cols: tuple[str, ...] = INFORMATIVE_NA_COLUMNS,
) -> pd.DataFrame:
    """Summarise missing values, classifying each column's NaN semantics.

    Args:
        df: Frame to analyse.
        informative_cols: Columns where NaN means "feature absent" per the
            data dictionary, as opposed to a genuinely unrecorded value.

    Returns:
        One row per column with missing values: ``n_missing``, ``pct_missing``,
        and ``kind`` ("informative" or "true gap"), sorted by ``pct_missing``.
    """
    n_missing = df.isna().sum()
    n_missing = n_missing[n_missing > 0]
    table = pd.DataFrame(
        {
            "n_missing": n_missing,
            "pct_missing": (n_missing / len(df) * 100).round(2),
            "kind": [
                "informative" if col in informative_cols else "true gap"
                for col in n_missing.index
            ],
        }
    )
    return table.sort_values("pct_missing", ascending=False)


def plot_missingness(table: pd.DataFrame, save_path: Path | None = None) -> Figure:
    """Horizontal bar chart of missingness, coloured by NaN semantics."""
    colors = table["kind"].map({"informative": "#4C72B0", "true gap": "#C44E52"})
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(table))))
    ax.barh(table.index, table["pct_missing"], color=colors)
    ax.set_xlabel("% missing")
    ax.set_title("Missingness by column — informative (blue) vs true gap (red)")
    ax.invert_yaxis()
    return _finalize(fig, save_path)


# ---------------------------------------------------------------------------
# Target distribution
# ---------------------------------------------------------------------------

def plot_target_distribution(
    y: pd.Series, save_path: Path | None = None
) -> Figure:
    """Raw vs log1p target histograms with skewness annotated on each."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, values, label in (
        (axes[0], y, f"{y.name} (raw)"),
        (axes[1], np.log1p(y), f"log1p({y.name})"),
    ):
        sns.histplot(values, kde=True, ax=ax)
        ax.set_title(f"{label} — skew = {pd.Series(values).skew():.2f}")
        ax.set_xlabel(label)
    return _finalize(fig, save_path)


# ---------------------------------------------------------------------------
# Skewness
# ---------------------------------------------------------------------------

def numeric_skewness(df: pd.DataFrame, threshold: float = 0.75) -> pd.Series:
    """Skewness of numeric columns exceeding ``threshold`` in magnitude.

    Returns:
        Skew values sorted by magnitude (descending). These are candidates
        for log/power transforms in preprocessing.
    """
    numeric = df.select_dtypes(include=np.number)
    skew = numeric.skew().dropna()
    skew = skew[skew.abs() > threshold]
    return skew.reindex(skew.abs().sort_values(ascending=False).index)


def plot_skewed_features(
    df: pd.DataFrame, columns: list[str], save_path: Path | None = None
) -> Figure:
    """Histogram grid of the given (typically most-skewed) numeric columns."""
    n_cols = 3
    n_rows = -(-len(columns) // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    for ax, col in zip(np.ravel(axes), columns):
        sns.histplot(df[col].dropna(), ax=ax, kde=True)
        ax.set_title(f"{col} — skew = {df[col].skew():.2f}", fontsize=9)
        ax.set_xlabel("")
    # Hide any unused panels in the grid.
    for ax in np.ravel(axes)[len(columns):]:
        ax.set_visible(False)
    return _finalize(fig, save_path)


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def target_correlations(df: pd.DataFrame, target: str) -> pd.Series:
    """Pearson correlation of every numeric feature with the target, sorted
    by magnitude (descending), target itself excluded."""
    corr = df.select_dtypes(include=np.number).corr()[target].drop(target)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def plot_correlation_heatmap(
    df: pd.DataFrame, target: str, top_n: int = 15, save_path: Path | None = None
) -> Figure:
    """Heatmap of the ``top_n`` numeric features most correlated with target."""
    top = target_correlations(df, target).head(top_n).index.tolist() + [target]
    corr = df[top].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax,
                annot_kws={"size": 7})
    ax.set_title(f"Correlation heatmap — top {top_n} features vs {target}")
    return _finalize(fig, save_path)


# ---------------------------------------------------------------------------
# Outliers & categorical structure
# ---------------------------------------------------------------------------

def plot_outlier_scatter(
    df: pd.DataFrame,
    x: str,
    target: str,
    save_path: Path | None = None,
) -> Figure:
    """Scatter of a feature vs target for visual outlier identification."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x, y=target, alpha=0.5, ax=ax)
    ax.set_title(f"{x} vs {target}")
    return _finalize(fig, save_path)


def plot_neighborhood_prices(
    df: pd.DataFrame, target: str, save_path: Path | None = None
) -> Figure:
    """Box plot of target by Neighborhood, ordered by median price."""
    order = df.groupby("Neighborhood")[target].median().sort_values().index
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="Neighborhood", y=target, order=order, ax=ax)
    ax.tick_params(axis="x", rotation=90)
    ax.set_title(f"{target} by Neighborhood (ordered by median)")
    return _finalize(fig, save_path)
