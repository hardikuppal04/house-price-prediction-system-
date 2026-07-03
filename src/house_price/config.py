"""Typed configuration loaded from ``config/config.yaml``.

The rest of the codebase depends only on these dataclasses, never on raw
dictionaries, so a typo in a config key fails loudly at load time rather than
silently deep inside a pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def project_root() -> Path:
    """Return the repository root (two levels above this file's package)."""
    # src/house_price/config.py -> src/house_price -> src -> <root>
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Filesystem locations, resolved to absolute paths under the repo root."""

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


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the project configuration.

    Args:
        path: Optional explicit path to a YAML config. Defaults to
            ``config/config.yaml`` under the repository root.

    Returns:
        A fully typed, immutable :class:`Config`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a required section or key is missing.
    """
    root = project_root()
    cfg_path = Path(path) if path is not None else root / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

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

    return Config(
        seed=int(raw["seed"]),
        paths=paths,
        dataset=dataset,
        split=split,
        cv=cv,
        target=target,
        preprocessing=preprocessing,
    )
