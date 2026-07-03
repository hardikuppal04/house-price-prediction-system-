"""Tests for the model zoo, CV harness plumbing, and ColumnSubset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from house_price.preprocessing import build_preprocessor
from house_price.training import ColumnSubset, build_model_zoo, evaluate_cv

CORE_MODELS = {
    "Linear",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "DecisionTree",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
}


def test_zoo_contains_core_models_and_records_skips() -> None:
    zoo = build_model_zoo(seed=42)
    assert CORE_MODELS <= set(zoo.models)
    # A model is either available or skipped with a reason — never silently gone.
    for name in ("XGBoost", "LightGBM", "CatBoost"):
        assert name in zoo.models or name in zoo.skipped
    assert all(isinstance(v, str) and v for v in zoo.skipped.values())


def test_zoo_models_are_seeded() -> None:
    zoo = build_model_zoo(seed=7)
    for name, model in zoo.models.items():
        params = model.get_params()
        seed_keys = [k for k in ("random_state", "random_seed") if k in params]
        if seed_keys:  # LinearRegression legitimately has no seed
            assert any(params[k] == 7 for k in seed_keys), f"{name} not seeded"


def test_column_subset_reindexes_with_zero_fill() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    sub = ColumnSubset(("a", "ghost")).fit(df)
    out = sub.transform(df)
    assert list(out.columns) == ["a", "ghost"]
    assert (out["ghost"] == 0).all()
    assert list(sub.get_feature_names_out()) == ["a", "ghost"]


def test_catboost_native_adapter_is_cloneable(synthetic_ames: pd.DataFrame) -> None:
    catboost = pytest.importorskip("catboost")  # noqa: F841
    from sklearn.base import clone

    from house_price.training import CatBoostNativeRegressor

    est = CatBoostNativeRegressor(iterations=20, random_seed=3)
    cloned = clone(est)  # the exact operation that broke raw CatBoostRegressor
    assert cloned.get_params()["random_seed"] == 3

    pre = build_preprocessor("raw", seed=3)
    X = synthetic_ames.drop(columns=["SalePrice"])
    y = np.log1p(synthetic_ames["SalePrice"])
    Xt = pre.fit_transform(X, y)
    cloned.fit(Xt, y)
    # Categorical columns were detected at fit time, not configured.
    assert "Neighborhood" in cloned.cat_features_
    assert np.isfinite(cloned.predict(Xt)).all()


def test_evaluate_cv_returns_full_metric_set(synthetic_ames: pd.DataFrame) -> None:
    X = synthetic_ames.drop(columns=["SalePrice"])
    y = np.log1p(synthetic_ames["SalePrice"])
    pipe = Pipeline([
        ("pre", build_preprocessor("ordinal", seed=0)),
        ("model", Ridge(alpha=1.0, random_state=0)),
    ])
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    metrics = evaluate_cv(pipe, X, y, cv)
    for key in ("log_rmse_mean", "log_rmse_std", "dollar_mae_mean",
                "dollar_mape_mean", "fit_time_s", "pred_latency_ms_per_row"):
        assert key in metrics
        assert np.isfinite(metrics[key])
    assert metrics["log_rmse_mean"] > 0
    assert metrics["fit_time_s"] > 0
