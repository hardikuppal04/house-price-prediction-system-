"""Metrics and result summaries.

The modelling target is log1p(SalePrice), so every evaluation reports two
views: log space (what models optimise) and dollar space (what stakeholders
understand), obtained by inverting the transform with expm1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


def adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
    """Adjusted R² — penalises feature count; NaN when n <= p + 1."""
    denom = n_samples - n_features - 1
    if denom <= 0:
        return float("nan")
    return 1.0 - (1.0 - r2) * (n_samples - 1) / denom


def regression_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    n_features: int | None = None,
) -> dict[str, float]:
    """Full metric set in both log space and dollar space.

    Args:
        y_true_log: True targets in log1p space.
        y_pred_log: Predictions in log1p space.
        n_features: Feature count for Adjusted R² (omitted -> NaN).

    Returns:
        Flat dict with ``log_``- and ``dollar_``-prefixed metrics.
    """
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    r2_log = r2_score(y_true_log, y_pred_log)
    r2_dollar = r2_score(y_true, y_pred)
    n = len(y_true_log)
    p = n_features if n_features is not None else 0

    return {
        "log_rmse": root_mean_squared_error(y_true_log, y_pred_log),
        "log_mae": mean_absolute_error(y_true_log, y_pred_log),
        "log_mse": mean_squared_error(y_true_log, y_pred_log),
        "log_r2": r2_log,
        "log_adj_r2": adjusted_r2(r2_log, n, p) if n_features is not None else float("nan"),
        "dollar_rmse": root_mean_squared_error(y_true, y_pred),
        "dollar_mae": mean_absolute_error(y_true, y_pred),
        "dollar_mse": mean_squared_error(y_true, y_pred),
        "dollar_r2": r2_dollar,
        "dollar_mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def summarize_cv(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Turn a ``{run_name: metrics}`` mapping into a sorted comparison table."""
    table = pd.DataFrame(results).T
    return table.sort_values("log_rmse_mean" if "log_rmse_mean" in table.columns else "log_rmse")


def consensus_selection(
    keep_sets: dict[str, set[str]], all_features: list[str]
) -> pd.DataFrame:
    """Combine per-method keep-sets into a consensus vote.

    Rule (documented in the report): a feature is kept when at least half of
    the applicable methods voted for it. "Applicable" matters — SHAP is
    skipped gracefully if unavailable, and the rule adapts to however many
    signals actually ran.

    Args:
        keep_sets: Method name -> set of feature names that method keeps.
        all_features: The full feature universe.

    Returns:
        One row per feature: per-method boolean votes, ``votes`` count and
        ``keep`` (consensus decision), sorted by votes descending.
    """
    if not keep_sets:
        raise ValueError("consensus_selection needs at least one method's votes")
    table = pd.DataFrame(
        {method: [f in kept for f in all_features] for method, kept in keep_sets.items()},
        index=all_features,
    )
    table["votes"] = table.sum(axis=1)
    threshold = len(keep_sets) / 2
    table["keep"] = table["votes"] >= threshold
    return table.sort_values("votes", ascending=False)
