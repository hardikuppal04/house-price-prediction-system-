# House Price Prediction

Production-style house price prediction on the **Ames Housing** dataset (OpenML id 42165).

> Status: **Milestone 1 — scaffold, config, and data ingestion.** Later milestones
> (EDA, preprocessing, model zoo, tuning, explainability, API/UI, docs) build on this.

## Quickstart

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -e .            # installable package (src/house_price)
pip install -r requirements-dev.txt

python scripts/download_data.py    # fetch + cache raw data (gitignored)
python scripts/make_split.py       # deterministic 80/20 holdout split
pytest                             # run the test suite
```

## Layout

```
config/        single config.yaml (paths, seed, CV folds, split)
scripts/       download_data.py, make_split.py
src/house_price/  installable package (config, data, utils, ...)
tests/         pytest unit tests
data/ models/ mlruns/ reports/figures/   gitignored artifacts
```

Raw data is downloaded by script and **never committed**. Configuration is loaded
through a typed dataclass (`src/house_price/config.py`); the codebase never reads
raw config dicts.

## Design notes

- **80/20 holdout**, stratified on target deciles, fixed seed, saved to disk. The
  holdout is touched exactly once, at final evaluation.
- **Target** is modelled as `log1p(SalePrice)`; metrics are reported in both log and
  dollar space.
- **Schema guard**: every raw fetch verifies the original 79-column Ames names are
  present, so a differently-encoded OpenML version fails loudly instead of silently
  breaking feature engineering.
