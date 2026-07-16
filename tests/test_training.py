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


def test_normalize_params_repairs_json_roundtrip() -> None:
    from house_price.training import _normalize_params

    raw = {
        "model__alpha": "0.000483",
        "n_estimators": 800,
        "model__max_features": "sqrt",
        "subsample": 0.8,
    }
    clean = _normalize_params(raw)
    assert clean["alpha"] == pytest.approx(0.000483)  # prefix stripped, float parsed
    assert clean["n_estimators"] == 800  # native int untouched
    assert clean["max_features"] == "sqrt"  # genuine string preserved
    assert clean["subsample"] == 0.8


def test_build_final_pipeline_applies_params() -> None:
    from house_price.config import load_config
    from house_price.training import build_final_pipeline

    cfg = load_config()
    pipe = build_final_pipeline(cfg, "Ridge", {"alpha": 3.5})
    assert pipe.named_steps["model"].alpha == 3.5
    # Linear models get the skew-corrected preprocessing they were tuned with.
    assert "log1p_skewed" in pipe.named_steps["pre"].named_steps
    with pytest.raises(ValueError, match="unavailable"):
        build_final_pipeline(cfg, "NotAModel", {})


def test_final_artifact_roundtrip(synthetic_ames: pd.DataFrame, tmp_path) -> None:
    """joblib dump/load of a fitted full pipeline predicts on raw-schema rows."""
    import joblib
    from sklearn.linear_model import Ridge as _Ridge
    from sklearn.pipeline import Pipeline as _Pipeline

    X = synthetic_ames.drop(columns=["SalePrice"])
    y = np.log1p(synthetic_ames["SalePrice"])
    pipe = _Pipeline(
        [
            ("pre", build_preprocessor("ohe", seed=0, log1p_skewed=True)),
            ("model", _Ridge(alpha=1.0, random_state=0)),
        ]
    ).fit(X, y)

    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    loaded = joblib.load(path)
    # One call, raw schema in, prediction out — the deployment contract.
    pred = loaded.predict(X.head(3))
    np.testing.assert_allclose(pred, pipe.predict(X.head(3)))


def test_evaluate_cv_returns_full_metric_set(synthetic_ames: pd.DataFrame) -> None:
    X = synthetic_ames.drop(columns=["SalePrice"])
    y = np.log1p(synthetic_ames["SalePrice"])
    pipe = Pipeline(
        [
            ("pre", build_preprocessor("ordinal", seed=0)),
            ("model", Ridge(alpha=1.0, random_state=0)),
        ]
    )
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    metrics = evaluate_cv(pipe, X, y, cv)
    for key in (
        "log_rmse_mean",
        "log_rmse_std",
        "dollar_mae_mean",
        "dollar_mape_mean",
        "fit_time_s",
        "pred_latency_ms_per_row",
    ):
        assert key in metrics
        assert np.isfinite(metrics[key])
    assert metrics["log_rmse_mean"] > 0
    assert metrics["fit_time_s"] > 0
