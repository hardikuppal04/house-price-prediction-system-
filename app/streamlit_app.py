"""Streamlit UI generated from the shared production feature schema."""

from __future__ import annotations

import os

import requests
import streamlit as st

from house_price.schema import FEATURES

st.set_page_config(page_title="Historical Ames Price Estimator", page_icon="🏠")
st.title("Historical Ames House Price Estimator")
st.warning(
    "Portfolio simulation only. Estimates are approximately 2010 dollars—not current valuations."
)

payload = {}
with st.form("prediction"):
    columns = st.columns(2)
    for index, spec in enumerate(FEATURES):
        with columns[index % 2]:
            if spec.kind == "category":
                payload[spec.name] = st.text_input(spec.label, value=str(spec.default or ""))
            elif spec.kind == "integer":
                payload[spec.name] = st.number_input(
                    spec.label,
                    min_value=int(spec.minimum),
                    max_value=int(spec.maximum),
                    value=int(spec.default),
                    step=1,
                )
            else:
                payload[spec.name] = st.number_input(
                    spec.label,
                    min_value=float(spec.minimum),
                    max_value=float(spec.maximum),
                    value=float(spec.default),
                    step=10.0,
                )
    submitted = st.form_submit_button("Estimate historical price")

if submitted:
    try:
        response = requests.post(
            os.getenv("HOUSE_PRICE_API", "http://localhost:8000/v1/predict"),
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
        st.metric("Estimated 2010 price", f"${result['predicted_price_2010_usd']:,.0f}")
        st.caption(
            f"90% empirical interval: ${result['interval_90_low']:,.0f}–"
            f"${result['interval_90_high']:,.0f}"
        )
        st.metric(
            f"Inflation-adjusted reference ({result['inflation_reference_year']} dollars)",
            f"${result['inflation_adjusted_reference_usd']:,.0f}",
        )
        st.caption(
            "CPI-U purchasing-power interval: "
            f"${result['inflation_adjusted_interval_90_low']:,.0f}–"
            f"${result['inflation_adjusted_interval_90_high']:,.0f}. "
            f"Factor from 2010 dollars: {result['inflation_factor']:.3f}x."
        )
        for warning in result["warnings"]:
            st.info(warning)
    except requests.RequestException:
        st.error("The prediction service is unavailable or rejected the request.")
