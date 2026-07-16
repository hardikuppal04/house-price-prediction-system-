"""Contract and leakage tests for the temporal production path."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from house_price.data import expanding_year_folds, make_temporal_split
from house_price.drift import drift_report, population_stability_index
from house_price.inflation import CPI_U_REFERENCE, adjust_2010_dollars
from house_price.production import predict_one, select_model
from house_price.provenance import write_json
from house_price.schema import FEATURE_NAMES, PredictionRequest, curated_frame


def _complete_raw(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["BldgType"] = "1Fam"
    out["HouseStyle"] = "1Story"
    out["BedroomAbvGr"] = 3
    out["TotRmsAbvGrd"] = 6
    out["CentralAir"] = "Y"
    return out


def _payload() -> dict:
    return {
        "neighborhood": "NAmes",
        "zoning": "RL",
        "building_type": "1Fam",
        "house_style": "1Story",
        "overall_quality": 6,
        "overall_condition": 5,
        "living_area_sqft": 1500,
        "basement_area_sqft": 900,
        "lot_area_sqft": 9000,
        "garage_cars": 2,
        "garage_area_sqft": 450,
        "year_built": 1975,
        "year_remodeled": 1995,
        "full_bathrooms": 2,
        "half_bathrooms": 1,
        "bedrooms": 3,
        "rooms": 6,
        "fireplaces": 1,
        "kitchen_quality": "TA",
        "central_air": "Y",
        "valuation_year": 2010,
    }


def test_temporal_holdout_contains_only_2010(synthetic_ames: pd.DataFrame) -> None:
    development, holdout = make_temporal_split(synthetic_ames)
    assert development["YrSold"].max() < 2010
    assert set(holdout["YrSold"]) == {2010}
    assert len(development) + len(holdout) == len(synthetic_ames)


def test_expanding_folds_never_train_on_future() -> None:
    frame = pd.DataFrame({"valuation_year": np.repeat(range(2006, 2010), 3)})
    for train, validation in expanding_year_folds(frame):
        train_max = frame.iloc[train]["valuation_year"].max()
        validation_min = frame.iloc[validation]["valuation_year"].min()
        assert train_max < validation_min


def test_curated_contract_matches_public_names(synthetic_ames: pd.DataFrame) -> None:
    curated = curated_frame(_complete_raw(synthetic_ames), target="SalePrice")
    assert tuple(curated.drop(columns="SalePrice").columns) == FEATURE_NAMES


def test_request_rejects_extra_and_impossible_dates() -> None:
    payload = _payload()
    payload["secret"] = "ignored?"
    with pytest.raises(ValueError):
        PredictionRequest.model_validate(payload)
    payload = _payload()
    payload["year_built"] = 2000
    payload["year_remodeled"] = 1990
    with pytest.raises(ValueError, match="year_remodeled"):
        PredictionRequest.model_validate(payload)


def test_one_se_rule_prefers_simpler_model() -> None:
    comparison = pd.DataFrame(
        [
            {"model": "baseline", "fold": "summary", "log_rmse": 0.20, "log_rmse_se": 0.01},
            {"model": "ridge", "fold": "summary", "log_rmse": 0.11, "log_rmse_se": 0.01},
            {"model": "elastic_net", "fold": "summary", "log_rmse": 0.105, "log_rmse_se": 0.01},
            {
                "model": "gradient_boosting",
                "fold": "summary",
                "log_rmse": 0.10,
                "log_rmse_se": 0.02,
            },
        ]
    )
    chosen, decision = select_model(comparison)
    assert chosen == "ridge"
    assert decision["baseline_improvement"] >= 0.20


def test_selection_falls_back_when_improvement_gate_fails() -> None:
    comparison = pd.DataFrame(
        [
            {"model": "baseline", "fold": "summary", "log_rmse": 0.20, "log_rmse_se": 0.01},
            {"model": "ridge", "fold": "summary", "log_rmse": 0.17, "log_rmse_se": 0.01},
        ]
    )
    assert select_model(comparison)[0] == "baseline"


class _ConstantModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        assert tuple(frame.columns) == FEATURE_NAMES
        return np.array([np.log1p(200_000)])


def test_bias_correction_interval_and_warning() -> None:
    metadata = {
        "smearing_factor": 1.02,
        "residual_log_quantiles": {"0.05": -0.1, "0.95": 0.1},
        "prediction_bounds": {"minimum": 50_000, "maximum": 500_000},
    }
    prediction = predict_one(_ConstantModel(), metadata, _payload())
    assert prediction.interval_low < prediction.price < prediction.interval_high
    assert prediction.price > 200_000
    assert "2010 dollars" in prediction.warnings[0]


def test_strict_json_converts_nonfinite_to_null(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    write_json(path, {"adjusted_r2": float("nan")})
    assert json.loads(path.read_text()) == {"adjusted_r2": None}
    assert "NaN" not in path.read_text()


def test_drift_reports_shift_and_novel_categories() -> None:
    reference = pd.DataFrame([_payload() for _ in range(20)])
    current = reference.copy()
    current["living_area_sqft"] = np.linspace(2500, 4000, len(current))
    current.loc[0, "neighborhood"] = "NovelPlace"
    report = drift_report(reference, current)
    assert report["numeric"]["living_area_sqft"]["psi"] > 0
    assert report["categorical"]["neighborhood"]["novel_category_rate"] > 0
    assert population_stability_index(reference["garage_cars"], current["garage_cars"]) == 0


def test_inflation_reference_is_purchasing_power_only() -> None:
    adjusted = adjust_2010_dollars(100_000)
    assert adjusted == pytest.approx(100_000 * CPI_U_REFERENCE.factor)
    assert CPI_U_REFERENCE.reference_year == 2025
    assert "not a present-day market valuation" in CPI_U_REFERENCE.note
