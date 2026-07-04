"""Model zoo, CV harness, MLflow tracking, and M4 experiments.

Boosters (XGBoost/LightGBM/CatBoost) are optional imports: a missing wheel on
this Python version degrades the zoo gracefully — the model is skipped with a
recorded reason, never a crash.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from house_price.config import Config
from house_price.data import load_split
from house_price.evaluation import consensus_selection, regression_metrics
from house_price.preprocessing import Encoding, build_preprocessor, drop_partial_sale_outliers
from house_price.utils import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - environment dependent
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    mlflow = None
    _MLFLOW_AVAILABLE = False


@dataclass
class ModelZoo:
    """Available estimators plus the record of what could not be loaded."""

    models: dict[str, BaseEstimator] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


def build_model_zoo(seed: int) -> ModelZoo:
    """Instantiate the full model list with fixed seeds.

    Baseline (untuned) settings on purpose — the zoo stage ranks model
    families; hyperparameter tuning happens in M5 on the shortlist.
    """
    zoo = ModelZoo()
    zoo.models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=10.0, random_state=seed),
        "Lasso": Lasso(alpha=0.001, random_state=seed, max_iter=50_000),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=seed, max_iter=50_000),
        "DecisionTree": DecisionTreeRegressor(random_state=seed),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=seed),
    }

    try:
        from xgboost import XGBRegressor

        zoo.models["XGBoost"] = XGBRegressor(
            n_estimators=500, learning_rate=0.05, random_state=seed, n_jobs=-1, verbosity=0
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        zoo.skipped["XGBoost"] = str(exc)
    try:
        from lightgbm import LGBMRegressor

        zoo.models["LightGBM"] = LGBMRegressor(
            n_estimators=500, learning_rate=0.05, random_state=seed, n_jobs=-1, verbose=-1
        )
    except ImportError as exc:  # pragma: no cover
        zoo.skipped["LightGBM"] = str(exc)
    try:
        from catboost import CatBoostRegressor

        zoo.models["CatBoost"] = CatBoostRegressor(
            iterations=500, learning_rate=0.05, random_seed=seed,
            verbose=0, allow_writing_files=False,
        )
    except ImportError as exc:  # pragma: no cover
        zoo.skipped["CatBoost"] = str(exc)

    for name, reason in zoo.skipped.items():
        logger.warning("Model skipped (unavailable): %s — %s", name, reason)
    return zoo


class CatBoostNativeRegressor(BaseEstimator, RegressorMixin):
    """CatBoost with native categorical handling, sklearn-clone-safe.

    Passing ``cat_features`` to CatBoost's constructor breaks sklearn's
    ``clone`` contract (the constructor mutates the parameter). Detecting the
    categorical columns at *fit time* from dtypes fixes that — and is more
    correct anyway: the column set is a property of the data reaching the
    model, not frozen configuration.
    """

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        random_seed: int = 42,
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_seed = random_seed

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CatBoostNativeRegressor":
        from catboost import CatBoostRegressor

        cat_cols = list(X.select_dtypes(exclude=np.number).columns)
        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            random_seed=self.random_seed,
            verbose=0,
            allow_writing_files=False,
        )
        self.model_.fit(X, y, cat_features=cat_cols)
        self.cat_features_ = cat_cols
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.predict(X)


class ColumnSubset(BaseEstimator, TransformerMixin):
    """Keep only the named (encoded) columns, tolerating fold-to-fold drift.

    After feature selection, CV refits may produce slightly different OHE
    columns per fold; reindexing with fill_value=0 keeps the schema stable
    (an absent dummy column is semantically "category not present").
    """

    def __init__(self, columns: tuple[str, ...]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ColumnSubset":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        return X.reindex(columns=list(self.columns), fill_value=0)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(self.columns, dtype=object)


# ---------------------------------------------------------------------------
# Data & CV harness
# ---------------------------------------------------------------------------

def load_training_data(cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Load the train split, apply the configured outlier policy, log the target."""
    train_df, _ = load_split(cfg)
    if cfg.preprocessing.drop_partial_sale_outliers:
        train_df = drop_partial_sale_outliers(train_df, cfg.dataset.target)
    X = train_df.drop(columns=[cfg.dataset.target])
    y = np.log1p(train_df[cfg.dataset.target])
    return X, y


def make_cv(cfg: Config) -> KFold:
    return KFold(n_splits=cfg.cv.n_folds, shuffle=cfg.cv.shuffle, random_state=cfg.seed)


def evaluate_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: KFold,
) -> dict[str, float]:
    """Manual CV loop returning metric means/stds plus timing.

    A manual loop (vs ``cross_validate``) lets one pass produce log- and
    dollar-space metrics, per-fold fit time, and prediction latency without
    a battery of custom scorers.
    """
    fold_metrics: list[dict[str, float]] = []
    fit_times: list[float] = []
    pred_times: list[float] = []

    for train_idx, val_idx in cv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = clone(pipeline)

        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pred = model.predict(X_val)
        pred_times.append((time.perf_counter() - t0) / len(X_val))

        fold_metrics.append(regression_metrics(y_val.to_numpy(), pred))

    frame = pd.DataFrame(fold_metrics)
    out: dict[str, float] = {}
    for col in frame.columns:
        out[f"{col}_mean"] = float(frame[col].mean())
        out[f"{col}_std"] = float(frame[col].std())
    out["fit_time_s"] = float(np.mean(fit_times))
    out["pred_latency_ms_per_row"] = float(np.mean(pred_times) * 1000)
    return out


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

def setup_mlflow(cfg: Config, experiment: str = "house-price-prediction") -> bool:
    """Point MLflow at a local SQLite store (gitignored); returns availability.

    MLflow >=3.14 deprecates the plain filesystem backend; a SQLite file under
    mlruns/ keeps tracking local and uncommitted while staying on the
    supported path.
    """
    if not _MLFLOW_AVAILABLE:
        logger.warning("MLflow unavailable — runs will not be tracked.")
        return False
    cfg.paths.mlruns.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(cfg.paths.mlruns / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(experiment)
    return True


def log_run(run_name: str, params: dict, metrics: dict[str, float]) -> None:
    """Log one run to MLflow; a tracking failure warns instead of crashing."""
    if not _MLFLOW_AVAILABLE:
        return
    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})
    except Exception as exc:  # noqa: BLE001 - tracking must never kill training
        logger.warning("MLflow logging failed for %s: %s", run_name, exc)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

@dataclass
class TuneResult:
    """Outcome of one tuning run, for the strategy comparison table."""

    model: str
    method: str
    best_score: float  # CV log-RMSE (lower is better)
    n_evals: int
    wall_time_s: float
    best_params: dict

    def as_row(self) -> dict:
        return {
            "model": self.model,
            "method": self.method,
            "cv_log_rmse": self.best_score,
            "n_evals": self.n_evals,
            "wall_time_s": round(self.wall_time_s, 1),
            "best_params": str(self.best_params),
        }


def _linear_pipe(model: BaseEstimator, seed: int) -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor("ohe", seed=seed, log1p_skewed=True)),
        ("model", model),
    ])


def _tree_pipe(model: BaseEstimator, seed: int) -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor("ohe", seed=seed)),
        ("model", model),
    ])


def tune_linear_grid(cfg: Config, X: pd.DataFrame, y: pd.Series) -> list[TuneResult]:
    """GridSearch for linear models — small convex spaces, exhaustive is cheap."""
    cv = make_cv(cfg)
    seed = cfg.seed
    searches = {
        "Ridge": (
            Ridge(random_state=seed),
            {"model__alpha": np.logspace(-2, 3, 20)},
        ),
        "Lasso": (
            Lasso(random_state=seed, max_iter=50_000),
            {"model__alpha": np.logspace(-5, -1, 20)},
        ),
        "ElasticNet": (
            ElasticNet(random_state=seed, max_iter=50_000),
            {
                "model__alpha": np.logspace(-5, -1, 10),
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        ),
    }
    results = []
    for name, (model, grid) in searches.items():
        logger.info("GridSearch: %s", name)
        t0 = time.perf_counter()
        gs = GridSearchCV(
            _linear_pipe(model, seed), grid, cv=cv,
            scoring="neg_root_mean_squared_error", n_jobs=-1, refit=False,
        ).fit(X, y)
        n_evals = len(gs.cv_results_["params"])
        res = TuneResult(name, "grid", -gs.best_score_, n_evals,
                         time.perf_counter() - t0, gs.best_params_)
        results.append(res)
        log_run(f"tune__{name}", {"model": name, "method": "grid", "stage": "tuning",
                                  **{k: str(v) for k, v in gs.best_params_.items()}},
                {"cv_log_rmse": res.best_score, "wall_time_s": res.wall_time_s})
    return results


def tune_forest_random(cfg: Config, X: pd.DataFrame, y: pd.Series,
                       n_iter: int = 25) -> TuneResult:
    """RandomizedSearch for the bagged-tree representative (large discrete space)."""
    cv = make_cv(cfg)
    seed = cfg.seed
    space = {
        "model__n_estimators": [200, 300, 500, 800],
        "model__max_depth": [None, 10, 15, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.3, 0.5, 1.0],
    }
    logger.info("RandomizedSearch: RandomForest (%d iters)", n_iter)
    t0 = time.perf_counter()
    rs = RandomizedSearchCV(
        _tree_pipe(RandomForestRegressor(random_state=seed, n_jobs=-1), seed),
        space, n_iter=n_iter, cv=cv, scoring="neg_root_mean_squared_error",
        random_state=seed, n_jobs=1, refit=False,
    ).fit(X, y)
    res = TuneResult("RandomForest", "random", -rs.best_score_, n_iter,
                     time.perf_counter() - t0, rs.best_params_)
    log_run("tune__RandomForest", {"model": "RandomForest", "method": "random",
                                   "stage": "tuning",
                                   **{k: str(v) for k, v in rs.best_params_.items()}},
            {"cv_log_rmse": res.best_score, "wall_time_s": res.wall_time_s})
    return res


def tune_boosters_optuna(cfg: Config, X: pd.DataFrame, y: pd.Series,
                         n_trials: int = 25) -> list[TuneResult]:
    """Optuna TPE for gradient boosters — continuous spaces, expensive evals.

    Search n_jobs stays at 1: the boosters parallelise internally and
    oversubscription would corrupt the wall-clock comparison.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = make_cv(cfg)
    seed = cfg.seed
    zoo = build_model_zoo(seed)

    def make_objective(name: str):
        def objective(trial: "optuna.Trial") -> float:
            if name == "LightGBM":
                from lightgbm import LGBMRegressor

                model = LGBMRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 300, 2000),
                    learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 15, 63),
                    min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                    random_state=seed, n_jobs=-1, verbose=-1,
                )
            elif name == "XGBoost":
                from xgboost import XGBRegressor

                model = XGBRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 300, 2000),
                    learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                    max_depth=trial.suggest_int("max_depth", 3, 8),
                    min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                    random_state=seed, n_jobs=-1, verbosity=0,
                )
            else:  # CatBoost
                from catboost import CatBoostRegressor

                model = CatBoostRegressor(
                    iterations=trial.suggest_int("iterations", 300, 1500),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    depth=trial.suggest_int("depth", 4, 8),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                    random_seed=seed, verbose=0, allow_writing_files=False,
                )
            score = cross_val_score(_tree_pipe(model, seed), X, y, cv=cv,
                                    scoring="neg_root_mean_squared_error", n_jobs=1)
            return -float(score.mean())

        return objective

    results = []
    for name in ("LightGBM", "XGBoost", "CatBoost"):
        if name not in zoo.models:
            logger.warning("Optuna: %s unavailable, skipped.", name)
            continue
        logger.info("Optuna TPE: %s (%d trials)", name, n_trials)
        t0 = time.perf_counter()
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(make_objective(name), n_trials=n_trials, show_progress_bar=False)
        res = TuneResult(name, "optuna_tpe", study.best_value, n_trials,
                         time.perf_counter() - t0, study.best_params)
        results.append(res)
        log_run(f"tune__{name}", {"model": name, "method": "optuna_tpe", "stage": "tuning",
                                  **{k: str(v) for k, v in study.best_params.items()}},
                {"cv_log_rmse": res.best_score, "wall_time_s": res.wall_time_s})
    return results


def _catboost_native_pipeline(seed: int) -> Pipeline | None:
    """CatBoost consuming raw categoricals — its native handling as a
    distinct encoding-comparison point. None if CatBoost is unavailable."""
    try:
        import catboost  # noqa: F401 - availability probe
    except ImportError:
        return None
    return Pipeline([
        ("pre", build_preprocessor("raw", seed=seed)),
        ("model", CatBoostNativeRegressor(random_seed=seed)),
    ])


def _normalize_params(params: dict) -> dict:
    """Undo JSON round-trip damage on tuned params.

    GridSearch keys carry the ``model__`` pipeline prefix, and numpy floats
    get stringified by ``json.dumps(default=str)``. Strip the prefix and
    parse numeric strings back to floats; genuine strings (e.g.
    ``max_features="sqrt"``) pass through unchanged.
    """
    clean: dict = {}
    for key, value in params.items():
        key = key.removeprefix("model__")
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        clean[key] = value
    return clean


def build_final_pipeline(cfg: Config, model_name: str, params: dict) -> Pipeline:
    """Reconstruct the winning pipeline from a tuned configuration.

    ``params`` may come straight from best_configs.json; linear models get
    the skew-corrected preprocessing they were tuned with.
    """
    zoo = build_model_zoo(cfg.seed)
    if model_name not in zoo.models:
        raise ValueError(f"Model {model_name!r} unavailable: {zoo.skipped.get(model_name)}")
    model = clone(zoo.models[model_name]).set_params(**_normalize_params(params))
    linear = model_name in ("Linear", "Ridge", "Lasso", "ElasticNet")
    builder = _linear_pipe if linear else _tree_pipe
    return builder(model, cfg.seed)


def finalize_model(cfg: Config, model_name: str, params: dict,
                   cv_log_rmse: float) -> Path:
    """Fit the winning pipeline on ALL training data and persist the artifact.

    The saved object is the entire pipeline (preprocessing included), so
    inference is a single ``joblib.load`` + ``predict`` on raw-schema rows.
    """
    import json

    import joblib

    X, y = load_training_data(cfg)
    pipe = build_final_pipeline(cfg, model_name, params)
    logger.info("Fitting final %s on %d training rows...", model_name, len(X))
    pipe.fit(X, y)

    cfg.paths.models.mkdir(parents=True, exist_ok=True)
    artifact = cfg.paths.models / "final_model.joblib"
    joblib.dump(pipe, artifact)

    meta = {
        "model": model_name,
        "params": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                   for k, v in params.items()},
        "encoding": "ohe",
        "target_transform": cfg.target.transform,
        "cv_log_rmse": cv_log_rmse,
        "n_train_rows": int(len(X)),
        "seed": cfg.seed,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (cfg.paths.models / "final_model_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    logger.info("Saved final pipeline to %s", artifact)
    return artifact


def evaluate_holdout(cfg: Config) -> dict[str, float]:
    """THE single holdout evaluation. Loads the frozen artifact, scores the
    292 never-seen rows, and persists predictions for M6 error analysis so
    no later stage needs to touch the model or holdout again."""
    import json

    import joblib

    artifact = cfg.paths.models / "final_model.joblib"
    pipe = joblib.load(artifact)

    _, holdout_df = load_split(cfg)  # outlier filter is train-only, by design
    X_hold = holdout_df.drop(columns=[cfg.dataset.target])
    y_hold_log = np.log1p(holdout_df[cfg.dataset.target])

    t0 = time.perf_counter()
    pred_log = pipe.predict(X_hold)
    latency_ms = (time.perf_counter() - t0) / len(X_hold) * 1000

    n_features = len(pipe[:-1].get_feature_names_out())
    metrics = regression_metrics(y_hold_log.to_numpy(), pred_log, n_features=n_features)
    metrics["pred_latency_ms_per_row"] = latency_ms

    out = holdout_df.copy()
    out["pred_log"] = pred_log
    out["pred_price"] = np.expm1(pred_log)
    out.to_parquet(cfg.paths.data_processed / "holdout_predictions.parquet", index=False)

    (cfg.paths.reports / "holdout_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in metrics.items()}, indent=2), encoding="utf-8"
    )
    log_run("holdout_final", {"stage": "holdout"}, metrics)
    logger.info("Holdout metrics: %s",
                {k: round(v, 5) for k, v in metrics.items()})
    return metrics


def run_encoding_experiment(cfg: Config) -> pd.DataFrame:
    """4-way encoding comparison on two representative models (+ CatBoost native).

    Ridge is the encoding-sensitive linear representative; LightGBM (or
    GradientBoosting as fallback) the encoding-tolerant tree representative.
    CatBoost native-vs-preencoded isolates the value of its own handling.
    """
    X, y = load_training_data(cfg)
    cv = make_cv(cfg)
    zoo = build_model_zoo(cfg.seed)
    tree_name = "LightGBM" if "LightGBM" in zoo.models else "GradientBoosting"
    representatives = {"Ridge": zoo.models["Ridge"], tree_name: zoo.models[tree_name]}

    results: dict[str, dict[str, float]] = {}
    for encoding in ("ohe", "ordinal", "target"):
        for name, model in representatives.items():
            pipe = Pipeline([
                ("pre", build_preprocessor(encoding, seed=cfg.seed,
                                           log1p_skewed=(name == "Ridge"))),
                ("model", clone(model)),
            ])
            run = f"{name}__{encoding}"
            logger.info("Encoding experiment: %s", run)
            metrics = evaluate_cv(pipe, X, y, cv)
            results[run] = metrics
            log_run(run, {"model": name, "encoding": encoding, "stage": "encoding"}, metrics)

    if "CatBoost" in zoo.models:
        for run, pipe in {
            "CatBoost__native": _catboost_native_pipeline(cfg.seed),
            "CatBoost__target": Pipeline([
                ("pre", build_preprocessor("target", seed=cfg.seed)),
                ("model", clone(zoo.models["CatBoost"])),
            ]),
        }.items():
            if pipe is None:
                continue
            logger.info("Encoding experiment: %s", run)
            metrics = evaluate_cv(pipe, X, y, cv)
            results[run] = metrics
            log_run(run, {"model": "CatBoost", "encoding": run.split("__")[1],
                          "stage": "encoding"}, metrics)

    table = pd.DataFrame(results).T.sort_values("log_rmse_mean")
    return table


def run_model_zoo(cfg: Config, encoding: Encoding) -> pd.DataFrame:
    """CV-evaluate every available model on the chosen encoding."""
    X, y = load_training_data(cfg)
    cv = make_cv(cfg)
    zoo = build_model_zoo(cfg.seed)

    results: dict[str, dict[str, float]] = {}
    for name, model in zoo.models.items():
        linear = name in ("Linear", "Ridge", "Lasso", "ElasticNet")
        pipe = Pipeline([
            ("pre", build_preprocessor(encoding, seed=cfg.seed, log1p_skewed=linear)),
            ("model", clone(model)),
        ])
        logger.info("Zoo: %s (%s)", name, encoding)
        metrics = evaluate_cv(pipe, X, y, cv)
        results[name] = metrics
        log_run(f"zoo__{name}", {"model": name, "encoding": encoding, "stage": "zoo"}, metrics)

    table = pd.DataFrame(results).T.sort_values("log_rmse_mean")
    return table


def run_feature_selection(
    cfg: Config, encoding: Encoding, top_k: int = 60
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Six-signal feature selection with a consensus vote, then CV validation.

    Honest caveat, also documented in the report: signals are computed once on
    the full training split (not nested inside each CV fold), so the follow-up
    CV comparison mildly favours the selected set. The decision this feeds is
    coarse (keep ~top-k features or not); the untouched holdout remains the
    final arbiter.

    Returns:
        ``(votes, comparison)`` — per-feature votes/keep decision, and the
        full-vs-selected CV comparison table.
    """
    X, y = load_training_data(cfg)
    cv = make_cv(cfg)
    seed = cfg.seed

    pre = build_preprocessor(encoding, seed=seed)
    Xt = pre.fit_transform(X, y)
    features = list(Xt.columns)
    logger.info("Selection universe: %d encoded features (top_k=%d)", len(features), top_k)

    def top(scores: pd.Series) -> set[str]:
        return set(scores.sort_values(ascending=False).head(top_k).index)

    keep_sets: dict[str, set[str]] = {}
    keep_sets["correlation"] = top(Xt.corrwith(y).abs())
    mi = mutual_info_regression(Xt, y, random_state=seed)
    keep_sets["mutual_info"] = top(pd.Series(mi, index=features))

    rfe = RFE(Ridge(alpha=10.0), n_features_to_select=top_k, step=0.1).fit(Xt, y)
    keep_sets["rfe"] = set(np.asarray(features)[rfe.support_])

    rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1).fit(Xt, y)
    keep_sets["tree_importance"] = top(pd.Series(rf.feature_importances_, index=features))

    # Permutation importance needs data the model was not fit on: an inner
    # subsplit of train (the holdout stays untouched until M5).
    X_in, X_val, y_in, y_val = train_test_split(Xt, y, test_size=0.25, random_state=seed)
    rf_in = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1).fit(X_in, y_in)
    perm = permutation_importance(rf_in, X_val, y_val, n_repeats=5, random_state=seed, n_jobs=-1)
    keep_sets["permutation"] = top(pd.Series(perm.importances_mean, index=features))

    try:
        import shap

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(Xt)
        keep_sets["shap"] = top(pd.Series(np.abs(shap_values).mean(axis=0), index=features))
    except Exception as exc:  # noqa: BLE001 - optional signal, degrade gracefully
        logger.warning("SHAP signal skipped: %s", exc)

    votes = consensus_selection(keep_sets, features)
    selected = tuple(votes.index[votes["keep"]])
    logger.info("Consensus kept %d/%d features (%d methods).",
                len(selected), len(features), len(keep_sets))

    comparison: dict[str, dict[str, float]] = {}
    zoo = build_model_zoo(seed)
    tree_name = "LightGBM" if "LightGBM" in zoo.models else "GradientBoosting"
    for name in ("Ridge", tree_name):
        for label, extra in (("full", None), ("selected", ColumnSubset(selected))):
            steps = [("pre", build_preprocessor(encoding, seed=seed,
                                                log1p_skewed=(name == "Ridge")))]
            if extra is not None:
                steps.append(("subset", extra))
            steps.append(("model", clone(zoo.models[name])))
            run = f"{name}__{label}"
            logger.info("Selection eval: %s", run)
            metrics = evaluate_cv(Pipeline(steps), X, y, cv)
            comparison[run] = metrics
            log_run(f"selection__{run}",
                    {"model": name, "features": label, "n_features":
                     len(selected) if extra else len(features), "stage": "selection"},
                    metrics)

    return votes, pd.DataFrame(comparison).T.sort_values("log_rmse_mean")
