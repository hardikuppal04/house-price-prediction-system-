"""Tests for the typed configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from house_price.config import CONFIG_ENV_VAR, Config, load_config, project_root


def _copy_config(target_root: Path) -> Path:
    target = target_root / "config" / "config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    source = project_root() / "config" / "config.yaml"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def test_load_default_config() -> None:
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.seed == 42
    assert cfg.dataset.name in {"ames", "california"}
    assert cfg.dataset.openml_id == 42165
    assert cfg.split.test_size == pytest.approx(0.2)
    assert cfg.cv.n_folds == 5
    assert cfg.target.transform == "log1p"
    assert isinstance(cfg.preprocessing.drop_partial_sale_outliers, bool)


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
    with pytest.raises(FileNotFoundError, match=CONFIG_ENV_VAR):
        load_config(tmp_path / "does_not_exist.yaml")


def test_cwd_config_resolves_paths_under_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_config(tmp_path)
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)

    cfg = load_config()

    assert cfg.paths.data_raw == tmp_path / "data" / "raw"
    assert cfg.paths.models == tmp_path / "models"
    assert cfg.paths.reports == tmp_path / "reports"


def test_env_config_overrides_default_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_root = tmp_path / "env_project"
    cfg_path = _copy_config(env_root)
    monkeypatch.setenv(CONFIG_ENV_VAR, str(cfg_path))
    monkeypatch.chdir(tmp_path)

    cfg = load_config()

    assert cfg.paths.data_processed == env_root / "data" / "processed"
    assert cfg.paths.mlruns == env_root / "mlruns"


def test_paths_ensure_creates_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config()
    # Redirect one path into tmp and confirm ensure() creates it.
    target = tmp_path / "nested" / "figures"
    object.__setattr__(cfg.paths, "figures", target)
    cfg.paths.ensure()
    assert target.exists()
