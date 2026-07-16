"""Temporal training, model selection, safe persistence, and inference."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from house_price.config import Config
from house_price.data import expanding_year_folds, make_temporal_split
from house_price.provenance import dependency_versions, git_commit, sha256_file, write_json
from house_price.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    NUMERIC_FEATURES,
    SCHEMA_VERSION,
    curated_frame,
    schema_document,
)

MODEL_FILENAME = "house_price_model.skops"
METADATA_FILENAME = "model_metadata.json"


@dataclass(frozen=True)
class Prediction:
    price: float
    interval_low: float
    interval_high: float
    log_prediction: float
    warnings: tuple[str, ...] = ()


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )


def candidate_models(seed: int) -> dict[str, Any]:
    return {
        "baseline": DummyRegressor(strategy="median"),
        "ridge": Ridge(alpha=10.0),
        "elastic_net": ElasticNet(alpha=0.0005, l1_ratio=0.8, max_iter=50_000, random_state=seed),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=2,
            loss="huber",
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=350,
            max_leaf_nodes=20,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }


def build_pipeline(model: Any) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def _fold_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    true_price = np.expm1(y_true_log)
    pred_price = np.expm1(y_pred_log)
    return {
        "log_rmse": float(root_mean_squared_error(y_true_log, y_pred_log)),
        "log_mae": float(mean_absolute_error(y_true_log, y_pred_log)),
        "dollar_mae": float(mean_absolute_error(true_price, pred_price)),
        "log_r2": float(r2_score(y_true_log, y_pred_log)),
    }


def temporal_model_comparison(
    X: pd.DataFrame,
    y_log: pd.Series,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    folds = expanding_year_folds(X)
    rows: list[dict[str, Any]] = []
    residuals: dict[str, list[np.ndarray]] = {}
    for name, estimator in candidate_models(seed).items():
        fold_values: list[float] = []
        residuals[name] = []
        started = time.perf_counter()
        for fold_number, (train_idx, validation_idx) in enumerate(folds, start=1):
            pipeline = build_pipeline(clone(estimator))
            pipeline.fit(X.iloc[train_idx], y_log.iloc[train_idx])
            predicted = pipeline.predict(X.iloc[validation_idx])
            metrics = _fold_metrics(y_log.iloc[validation_idx].to_numpy(), predicted)
            fold_values.append(metrics["log_rmse"])
            residuals[name].append(y_log.iloc[validation_idx].to_numpy() - predicted)
            rows.append(
                {
                    "model": name,
                    "fold": fold_number,
                    "validation_year": 2006 + fold_number,
                    **metrics,
                }
            )
        rows.append(
            {
                "model": name,
                "fold": "summary",
                "validation_year": None,
                "log_rmse": float(np.mean(fold_values)),
                "log_rmse_se": float(np.std(fold_values, ddof=1) / np.sqrt(len(fold_values))),
                "fit_time_s": time.perf_counter() - started,
            }
        )
    return pd.DataFrame(rows), {name: np.concatenate(values) for name, values in residuals.items()}


def select_model(comparison: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    summaries = comparison.loc[comparison["fold"] == "summary"].copy()
    best = summaries.loc[summaries["log_rmse"].idxmin()]
    threshold = float(best["log_rmse"] + best["log_rmse_se"])
    complexity = {
        "ridge": 1,
        "elastic_net": 2,
        "gradient_boosting": 3,
        "hist_gradient_boosting": 4,
    }
    eligible = summaries[
        (summaries["model"] != "baseline") & (summaries["log_rmse"] <= threshold)
    ].copy()
    baseline_rmse = float(summaries.loc[summaries["model"] == "baseline", "log_rmse"].iloc[0])
    best_model_rmse = float(summaries.loc[summaries["model"] != "baseline", "log_rmse"].min())
    improvement = 1.0 - best_model_rmse / baseline_rmse
    if improvement < 0.20 or eligible.empty:
        chosen = "baseline"
    else:
        chosen = min(eligible["model"], key=lambda name: complexity[name])
    return chosen, {
        "best_cv_model": str(best["model"]),
        "one_se_threshold": threshold,
        "baseline_improvement": improvement,
        "selection_rule": "least complex model within one SE; require 20% baseline improvement",
    }


def train_production_model(cfg: Config, raw: pd.DataFrame) -> dict[str, Any]:
    development_raw, _ = make_temporal_split(raw)
    development = curated_frame(development_raw, target=cfg.dataset.target)
    X = development.loc[:, FEATURE_NAMES]
    y_log = np.log1p(development[cfg.dataset.target])
    comparison, residuals = temporal_model_comparison(X, y_log, cfg.seed)
    chosen, decision = select_model(comparison)
    pipeline = build_pipeline(candidate_models(cfg.seed)[chosen])
    pipeline.fit(X, y_log)

    model_residuals = residuals[chosen]
    smearing = float(np.mean(np.exp(model_residuals)))
    q_low, q_high = np.quantile(model_residuals, [0.05, 0.95])
    price_min = float(max(0, development[cfg.dataset.target].quantile(0.001) * 0.5))
    price_max = float(development[cfg.dataset.target].quantile(0.999) * 1.5)

    cfg.paths.models.mkdir(parents=True, exist_ok=True)
    artifact_path = cfg.paths.models / MODEL_FILENAME
    _dump_skops(pipeline, artifact_path)
    comparison_path = cfg.paths.reports / "experiments" / "temporal_model_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)

    metadata = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "schema_version": SCHEMA_VERSION,
        "model": chosen,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_years": [2006, 2007, 2008, 2009],
        "holdout_year": 2010,
        "n_training_rows": len(X),
        "seed": cfg.seed,
        "smearing_factor": smearing,
        "residual_log_quantiles": {"0.05": float(q_low), "0.95": float(q_high)},
        "prediction_bounds": {"minimum": price_min, "maximum": price_max},
        "decision": decision,
        "artifact_sha256": sha256_file(artifact_path),
        "git_commit": git_commit(Path(__file__).resolve().parents[2]),
        "dependencies": dependency_versions(("numpy", "pandas", "scikit-learn", "skops")),
        "feature_schema": schema_document(),
        "limitations": [
            "Predicts historical Ames sale prices only; outputs are approximately 2010 dollars.",
            "Not valid for present-day pricing or locations outside Ames, Iowa.",
        ],
    }
    write_json(cfg.paths.models / METADATA_FILENAME, metadata)
    _log_mlflow(cfg, comparison, metadata)
    return metadata


def _dump_skops(model: Any, path: Path) -> None:
    try:
        import skops.io as sio
    except ImportError as exc:
        raise RuntimeError("Install skops to persist the production model") from exc
    sio.dump(model, path)


def load_production_model(models_dir: Path) -> tuple[Any, dict[str, Any]]:
    import json

    try:
        import skops.io as sio
    except ImportError as exc:
        raise RuntimeError("Install skops to load the production model") from exc
    artifact = models_dir / MODEL_FILENAME
    metadata_path = models_dir / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(artifact) != metadata["artifact_sha256"]:
        raise RuntimeError("Model artifact checksum does not match metadata")
    unknown = sio.get_untrusted_types(file=artifact)
    allowed_prefixes = ("sklearn.", "numpy.", "house_price.")
    rejected = [name for name in unknown if not name.startswith(allowed_prefixes)]
    if rejected:
        raise RuntimeError(f"Model contains unapproved types: {rejected}")
    return sio.load(artifact, trusted=unknown), metadata


def predict_one(model: Any, metadata: dict[str, Any], payload: dict[str, Any]) -> Prediction:
    frame = pd.DataFrame([{name: payload[name] for name in FEATURE_NAMES}])
    log_prediction = float(model.predict(frame)[0])
    smear = float(metadata["smearing_factor"])
    residuals = metadata["residual_log_quantiles"]
    low_log = log_prediction + float(residuals["0.05"])
    high_log = log_prediction + float(residuals["0.95"])
    bounds = metadata["prediction_bounds"]
    raw_price = np.exp(log_prediction) * smear - 1
    price = float(np.clip(raw_price, bounds["minimum"], bounds["maximum"]))
    low = float(np.clip(np.exp(low_log) * smear - 1, bounds["minimum"], bounds["maximum"]))
    high = float(np.clip(np.exp(high_log) * smear - 1, bounds["minimum"], bounds["maximum"]))
    warnings = ["Historical estimate in approximately 2010 dollars; not a present-day valuation."]
    if price != raw_price:
        warnings.append("Prediction was clipped to the validated training range.")
    return Prediction(price, low, high, log_prediction, tuple(warnings))


def _log_mlflow(cfg: Config, comparison: pd.DataFrame, metadata: dict[str, Any]) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{(cfg.paths.mlruns / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("house-price-temporal")
        with mlflow.start_run(run_name=f"production-{metadata['model_version']}"):
            mlflow.log_params(
                {
                    "model": metadata["model"],
                    "seed": metadata["seed"],
                    "schema_version": metadata["schema_version"],
                }
            )
            summaries = comparison[comparison["fold"] == "summary"].set_index("model")
            mlflow.log_metric("cv_log_rmse", float(summaries.loc[metadata["model"], "log_rmse"]))
    except Exception:
        # Tracking is an observability enhancement, never a training dependency.
        return
