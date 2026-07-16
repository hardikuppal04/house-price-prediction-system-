"""One-time temporal holdout evaluation and portfolio reports."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from house_price.config import Config
from house_price.data import make_temporal_split
from house_price.production import load_production_model, predict_one
from house_price.provenance import write_json
from house_price.schema import FEATURE_NAMES, curated_frame


def evaluate_final_holdout(
    cfg: Config,
    raw: pd.DataFrame,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate 2010 once; refuse accidental overwrite unless explicitly forced."""
    metrics_path = cfg.paths.reports / "temporal_holdout_metrics.json"
    if metrics_path.exists() and not force:
        raise FileExistsError(
            f"Final holdout already evaluated at {metrics_path}; "
            "use --force only for an intentional rerun"
        )
    _, holdout_raw = make_temporal_split(raw)
    holdout = curated_frame(holdout_raw, target=cfg.dataset.target)
    X = holdout.loc[:, FEATURE_NAMES]
    y_true = holdout[cfg.dataset.target].to_numpy(dtype=float)
    model, metadata = load_production_model(cfg.paths.models)
    predictions = [predict_one(model, metadata, row) for row in X.to_dict(orient="records")]
    y_pred = np.asarray([prediction.price for prediction in predictions])
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    covered = np.asarray(
        [
            prediction.interval_low <= truth <= prediction.interval_high
            for prediction, truth in zip(predictions, y_true)
        ]
    )
    metrics: dict[str, Any] = {
        "model_version": metadata["model_version"],
        "holdout_year": 2010,
        "n_rows": len(X),
        "log_rmse": float(root_mean_squared_error(log_true, log_pred)),
        "log_mae": float(mean_absolute_error(log_true, log_pred)),
        "dollar_rmse": float(root_mean_squared_error(y_true, y_pred)),
        "dollar_mae": float(mean_absolute_error(y_true, y_pred)),
        "dollar_mape": float(np.mean(np.abs((y_true - y_pred) / y_true))),
        "dollar_r2": float(r2_score(y_true, y_pred)),
        "interval_90_coverage": float(covered.mean()),
    }
    predictions_frame = X.copy()
    predictions_frame["actual_price"] = y_true
    predictions_frame["predicted_price"] = y_pred
    predictions_frame["residual"] = y_true - y_pred
    predictions_frame["interval_low"] = [p.interval_low for p in predictions]
    predictions_frame["interval_high"] = [p.interval_high for p in predictions]
    predictions_frame.to_parquet(
        cfg.paths.reports / "temporal_holdout_predictions.parquet", index=False
    )
    write_json(metrics_path, metrics)
    _write_error_reports(cfg, predictions_frame)
    _write_importance(cfg, model, X, log_true)
    return metrics


def _write_error_reports(cfg: Config, frame: pd.DataFrame) -> None:
    output = cfg.paths.reports / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    frame.assign(
        price_band=pd.qcut(frame["actual_price"], 4, duplicates="drop"),
    ).groupby("price_band", observed=True).agg(
        count=("actual_price", "size"),
        actual_mean=("actual_price", "mean"),
        prediction_mean=("predicted_price", "mean"),
        mae=("residual", lambda values: values.abs().mean()),
    ).to_csv(
        output / "errors_by_price_band.csv"
    )
    frame.groupby("neighborhood").agg(
        count=("actual_price", "size"),
        actual_mean=("actual_price", "mean"),
        prediction_mean=("predicted_price", "mean"),
        mae=("residual", lambda values: values.abs().mean()),
    ).sort_values("count", ascending=False).to_csv(output / "errors_by_neighborhood.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(frame["actual_price"], frame["predicted_price"], alpha=0.55)
    bounds = [
        frame[["actual_price", "predicted_price"]].min().min(),
        frame[["actual_price", "predicted_price"]].max().max(),
    ]
    axes[0].plot(bounds, bounds, "--", color="black")
    axes[0].set(xlabel="Actual 2010 price", ylabel="Predicted price", title="Calibration")
    axes[1].scatter(frame["predicted_price"], frame["residual"], alpha=0.55)
    axes[1].axhline(0, linestyle="--", color="black")
    axes[1].set(xlabel="Predicted price", ylabel="Actual - predicted", title="Residuals")
    fig.tight_layout()
    fig.savefig(output / "holdout_diagnostics.png", dpi=160)
    plt.close(fig)


def _write_importance(cfg: Config, model: Any, X: pd.DataFrame, y_log: np.ndarray) -> None:
    output = cfg.paths.reports / "analysis"
    result = permutation_importance(
        model,
        X,
        y_log,
        scoring="neg_root_mean_squared_error",
        n_repeats=10,
        random_state=cfg.seed,
        n_jobs=1,
    )
    pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).to_csv(
        output / "permutation_importance.csv", index=False
    )
    try:
        import shap  # noqa: F401

        shap_status = {
            "status": "available",
            "note": "Use permutation importance as model-agnostic default.",
        }
    except ImportError:
        shap_status = {"status": "skipped", "reason": "optional dependency not installed"}
    write_json(output / "shap_status.json", shap_status)
