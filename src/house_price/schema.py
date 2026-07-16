"""Single source of truth for the production prediction contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: str
    kind: Literal["integer", "number", "category"]
    label: str
    minimum: float | None = None
    maximum: float | None = None
    categories: tuple[str, ...] = ()
    default: Any = None


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("neighborhood", "Neighborhood", "category", "Neighborhood", default="NAmes"),
    FeatureSpec("zoning", "MSZoning", "category", "Zoning", default="RL"),
    FeatureSpec("building_type", "BldgType", "category", "Building type", default="1Fam"),
    FeatureSpec("house_style", "HouseStyle", "category", "House style", default="1Story"),
    FeatureSpec("overall_quality", "OverallQual", "integer", "Overall quality", 1, 10, default=5),
    FeatureSpec(
        "overall_condition", "OverallCond", "integer", "Overall condition", 1, 10, default=5
    ),
    FeatureSpec(
        "living_area_sqft", "GrLivArea", "number", "Living area (sq ft)", 200, 6000, default=1500
    ),
    FeatureSpec(
        "basement_area_sqft", "TotalBsmtSF", "number", "Basement area (sq ft)", 0, 6500, default=900
    ),
    FeatureSpec(
        "lot_area_sqft", "LotArea", "number", "Lot area (sq ft)", 1000, 250000, default=9000
    ),
    FeatureSpec("garage_cars", "GarageCars", "integer", "Garage capacity", 0, 5, default=2),
    FeatureSpec(
        "garage_area_sqft", "GarageArea", "number", "Garage area (sq ft)", 0, 1500, default=450
    ),
    FeatureSpec("year_built", "YearBuilt", "integer", "Year built", 1800, 2010, default=1975),
    FeatureSpec(
        "year_remodeled", "YearRemodAdd", "integer", "Year remodeled", 1800, 2010, default=1995
    ),
    FeatureSpec("full_bathrooms", "FullBath", "integer", "Full bathrooms", 0, 4, default=2),
    FeatureSpec("half_bathrooms", "HalfBath", "integer", "Half bathrooms", 0, 3, default=1),
    FeatureSpec("bedrooms", "BedroomAbvGr", "integer", "Bedrooms", 0, 8, default=3),
    FeatureSpec("rooms", "TotRmsAbvGrd", "integer", "Rooms", 1, 15, default=6),
    FeatureSpec("fireplaces", "Fireplaces", "integer", "Fireplaces", 0, 4, default=1),
    FeatureSpec("kitchen_quality", "KitchenQual", "category", "Kitchen quality", default="TA"),
    FeatureSpec("central_air", "CentralAir", "category", "Central air", default="Y"),
    FeatureSpec(
        "valuation_year", "YrSold", "integer", "Historical valuation year", 2006, 2010, default=2010
    ),
)

FEATURE_NAMES = tuple(feature.name for feature in FEATURES)
NUMERIC_FEATURES = tuple(f.name for f in FEATURES if f.kind != "category")
CATEGORICAL_FEATURES = tuple(f.name for f in FEATURES if f.kind == "category")


class PredictionRequestBase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @model_validator(mode="after")
    def validate_dates(self):
        year_built = getattr(self, "year_built")
        year_remodeled = getattr(self, "year_remodeled")
        valuation_year = getattr(self, "valuation_year")
        if year_built > valuation_year:
            raise ValueError("year_built cannot be after valuation_year")
        if year_remodeled < year_built:
            raise ValueError("year_remodeled cannot be before year_built")
        if year_remodeled > valuation_year:
            raise ValueError("year_remodeled cannot be after valuation_year")
        return self


def make_prediction_request_model() -> type[BaseModel]:
    """Generate the API model from the same specs used by training and the UI."""
    fields: dict[str, tuple[type, Any]] = {}
    for spec in FEATURES:
        annotation = str if spec.kind == "category" else (int if spec.kind == "integer" else float)
        constraints: dict[str, Any] = {"title": spec.label}
        if spec.kind == "category":
            constraints["min_length"] = 1
        else:
            constraints.update(ge=spec.minimum, le=spec.maximum)
        fields[spec.name] = (annotation, Field(..., **constraints))
    return create_model("PredictionRequest", __base__=PredictionRequestBase, **fields)  # type: ignore[call-overload]


PredictionRequest = make_prediction_request_model()


def curated_frame(raw: pd.DataFrame, *, target: str | None = None) -> pd.DataFrame:
    """Convert raw Ames columns to the stable public feature names."""
    missing = [f.source for f in FEATURES if f.source not in raw.columns]
    if missing:
        raise ValueError(f"Raw data is missing curated columns: {missing}")
    out = raw.loc[:, [f.source for f in FEATURES]].rename(
        columns={f.source: f.name for f in FEATURES}
    )
    if target is not None:
        if target not in raw.columns:
            raise ValueError(f"Raw data is missing target column {target!r}")
        out[target] = pd.to_numeric(raw[target], errors="raise")
    return out


def schema_document() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "features": [spec.__dict__ for spec in FEATURES],
    }
