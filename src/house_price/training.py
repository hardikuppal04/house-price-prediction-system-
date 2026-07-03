"""Model zoo, CV harness, MLflow tracking, and M4 experiments.

Boosters (XGBoost/LightGBM/CatBoost) are optional imports: a missing wheel on
this Python version degrades the zoo gracefully — the model is skipped with a
recorded reason, never a crash.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, train_test_split
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
