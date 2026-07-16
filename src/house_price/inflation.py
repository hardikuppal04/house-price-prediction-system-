"""Inflation-only dollar conversion for historical predictions.

This is intentionally not a house-market appreciation model. It only converts
the model's 2010-dollar estimate into a CPI-U purchasing-power reference.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InflationReference:
    base_year: int
    base_cpi_u: float
    reference_year: int
    reference_cpi_u: float
    source: str
    note: str

    @property
    def factor(self) -> float:
        return self.reference_cpi_u / self.base_cpi_u


CPI_U_REFERENCE = InflationReference(
    base_year=2010,
    base_cpi_u=218.056,
    reference_year=2025,
    reference_cpi_u=322.561,
    source="BLS CPI-U All items, U.S. city average, not seasonally adjusted",
    note=(
        "Inflation-adjusted purchasing-power reference only; not a present-day "
        "market valuation, appraisal, or forecast."
    ),
)


def adjust_2010_dollars(amount: float, reference: InflationReference = CPI_U_REFERENCE) -> float:
    """Convert a 2010-dollar amount into reference-year CPI-U dollars."""
    return amount * reference.factor
