"""Typed configuration loaded from ``config/config.yaml``.

The rest of the codebase depends only on these dataclasses, never on raw
dictionaries, so a typo in a config key fails loudly at load time rather than
silently deep inside a pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_ENV_VAR = "HOUSE_PRICE_CONFIG"


def project_root() -> Path:
    """Return the editable-source repository root."""
    # src/house_price/config.py -> src/house_price -> src -> <root>
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Filesystem locations, resolved under the selected project root."""

    data_raw: Path
    data_processed: Path
    models: Path
    reports: Path
    figures: Path
    mlruns: Path

    def ensure(self) -> None:
        """Create every configured directory if it does not already exist."""
        for path in (
            self.data_raw,
            self.data_processed,
            self.models,
            self.reports,
            self.figures,
            self.mlruns,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    openml_id: int
    openml_version: int
    target: str
    raw_filename: str


@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    stratify_bins: int
    train_filename: str
    holdout_filename: str


@dataclass(frozen=True)
class CVConfig:
    n_folds: int
    shuffle: bool


@dataclass(frozen=True)
class TargetConfig:
    transform: str


@dataclass(frozen=True)
class PreprocessingConfig:
    drop_partial_sale_outliers: bool


@dataclass(frozen=True)
class Config:
    """Top-level typed view of ``config.yaml``."""

    seed: int
    paths: Paths
    dataset: DatasetConfig
    split: SplitConfig
    cv: CVConfig
    target: TargetConfig
    preprocessing: PreprocessingConfig


def _resolve(root: Path, value: str) -> Path:
    """Resolve a possibly-relative config path against the repo root."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def _config_root(cfg_path: Path) -> Path:
    """Return the project root implied by a config file location."""
    parent = cfg_path.parent
    return parent.parent if parent.name == "config" else parent


def _default_config_candidates() -> list[Path]:
    return [
        Path.cwd() / "config" / "config.yaml",
        project_root() / "config" / "config.yaml",
    ]


def _select_config_path(path: str | Path | None) -> Path:
    if path is not None:
        cfg_path = Path(path).expanduser()
        return cfg_path if cfg_path.is_absolute() else Path.cwd() / cfg_path

    env_path = os.getenv(CONFIG_ENV_VAR)
    if env_path:
        cfg_path = Path(env_path).expanduser()
        return cfg_path if cfg_path.is_absolute() else Path.cwd() / cfg_path

    for candidate in _default_config_candidates():
        if candidate.exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in _default_config_candidates())
    raise FileNotFoundError(
        "Config file not found. Searched: "
        f"{searched}. Set {CONFIG_ENV_VAR}=C:\\path\\to\\config.yaml to override."
    )


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the project configuration.

    Args:
        path: Optional explicit path to a YAML config. Defaults to
            ``$HOUSE_PRICE_CONFIG``, ``./config/config.yaml``, then the
            editable-source checkout layout.

    Returns:
        A fully typed, immutable :class:`Config`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a required section or key is missing.
    """
    cfg_path = _select_config_path(path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {cfg_path}. "
            f"Set {CONFIG_ENV_VAR}=C:\\path\\to\\config.yaml to override."
        )
    root = _config_root(cfg_path)

    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    paths = Paths(
        data_raw=_resolve(root, raw["paths"]["data_raw"]),
        data_processed=_resolve(root, raw["paths"]["data_processed"]),
        models=_resolve(root, raw["paths"]["models"]),
        reports=_resolve(root, raw["paths"]["reports"]),
        figures=_resolve(root, raw["paths"]["figures"]),
        mlruns=_resolve(root, raw["paths"]["mlruns"]),
    )
    dataset = DatasetConfig(**raw["dataset"])
    split = SplitConfig(**raw["split"])
    cv = CVConfig(**raw["cv"])
    target = TargetConfig(**raw["target"])
    preprocessing = PreprocessingConfig(**raw["preprocessing"])

    config = Config(
        seed=int(raw["seed"]),
        paths=paths,
        dataset=dataset,
        split=split,
        cv=cv,
        target=target,
        preprocessing=preprocessing,
    )
    _validate_config(config)
    return config


def _validate_config(cfg: Config) -> None:
    """Fail early on invalid values instead of deferring errors to sklearn."""
    if cfg.dataset.name != "ames":
        raise ValueError("The production workflow supports only the Ames dataset")
    if not 0 < cfg.split.test_size < 1:
        raise ValueError("split.test_size must be between 0 and 1")
    if cfg.split.stratify_bins < 2:
        raise ValueError("split.stratify_bins must be at least 2")
    if cfg.cv.n_folds < 2:
        raise ValueError("cv.n_folds must be at least 2")
    if cfg.target.transform != "log1p":
        raise ValueError("Only the log1p target transform is supported")
