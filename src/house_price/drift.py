"""Offline drift and delayed-label performance reporting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from house_price.schema import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        reference_value = float(reference.iloc[0])
        current_outside_reference = ~np.isclose(current.to_numpy(), reference_value)
        if not current_outside_reference.any():
            return 0.0
        return float(current_outside_reference.mean())
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_counts = np.histogram(current, bins=edges)[0] / len(current)
    ref_counts = np.clip(ref_counts, 1e-6, None)
    cur_counts = np.clip(cur_counts, 1e-6, None)
    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {"numeric": {}, "categorical": {}, "range_violations": {}}
    for name in NUMERIC_FEATURES:
        report["numeric"][name] = {
            "psi": population_stability_index(reference[name], current[name]),
            "reference_missing_rate": float(reference[name].isna().mean()),
            "current_missing_rate": float(current[name].isna().mean()),
        }
    for name in CATEGORICAL_FEATURES:
        known = set(reference[name].dropna().astype(str))
        values = current[name].dropna().astype(str)
        report["categorical"][name] = {
            "novel_category_rate": float((~values.isin(known)).mean()) if len(values) else 0.0,
            "current_missing_rate": float(current[name].isna().mean()),
        }
    for spec in FEATURES:
        if spec.kind == "category":
            continue
        values = pd.to_numeric(current[spec.name], errors="coerce")
        invalid = ((values < spec.minimum) | (values > spec.maximum)).fillna(False)
        report["range_violations"][spec.name] = float(invalid.mean())
    if {"actual_price", "predicted_price"} <= set(current.columns):
        actual = current["actual_price"]
        predicted = current["predicted_price"]
        report["delayed_label_performance"] = {
            "log_rmse": float(root_mean_squared_error(np.log1p(actual), np.log1p(predicted))),
            "dollar_mae": float(mean_absolute_error(actual, predicted)),
        }
    if "predicted_price" in reference and "predicted_price" in current:
        report["prediction_shift"] = {
            "psi": population_stability_index(
                reference["predicted_price"], current["predicted_price"]
            ),
            "reference_mean": float(reference["predicted_price"].mean()),
            "current_mean": float(current["predicted_price"].mean()),
        }
    return report
