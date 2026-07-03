"""Tests for the typed configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from house_price.config import Config, load_config, project_root


def test_load_default_config() -> None:
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.seed == 42
    assert cfg.dataset.name in {"ames", "california"}
    assert cfg.dataset.openml_id == 42165
    assert cfg.split.test_size == pytest.approx(0.2)
    assert cfg.cv.n_folds == 5
    assert cfg.target.transform == "log1p"


def test_paths_are_absolute_under_root() -> None:
    cfg = load_config()
    root = project_root()
    for path in (
        cfg.paths.data_raw,
        cfg.paths.data_processed,
        cfg.paths.models,
        cfg.paths.figures,
        cfg.paths.mlruns,
    ):
        assert path.is_absolute()
        assert str(path).startswith(str(root))


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_paths_ensure_creates_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config()
    # Redirect one path into tmp and confirm ensure() creates it.
    target = tmp_path / "nested" / "figures"
    object.__setattr__(cfg.paths, "figures", target)
    cfg.paths.ensure()
    assert target.exists()
