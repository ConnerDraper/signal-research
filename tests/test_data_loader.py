#!/usr/bin/env python3
"""Tests for data_loader.py"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from src.data_loader import apply_filters, prepare_data, validate_data


def test_prepare_data_converts_to_decimal(barra_data: pl.DataFrame):
    """Test that returns and risk are correctly converted to decimals."""
    config = {"convert_returns_to_decimal": True, "replace_zero_volume": False}
    result = prepare_data(barra_data, config)

    expected = barra_data.with_columns(
        pl.col("return") / 100,
        pl.col("specific_return") / 100,
        pl.col("specific_risk") / 100,
    )
    assert_frame_equal(result, expected)


def test_prepare_data_replaces_zero_volume(barra_data: pl.DataFrame):
    """Test that zero volume is correctly replaced with null."""
    config = {"convert_returns_to_decimal": False, "replace_zero_volume": True}
    result = prepare_data(barra_data, config)

    expected = barra_data.with_columns(pl.col("daily_volume").replace(0, None))
    assert_frame_equal(result, expected)


def test_apply_filters_usa_only(barra_data: pl.DataFrame):
    """Test the USA-only filter."""
    filters = [{"usa_only": True}]
    result = apply_filters(barra_data, filters)
    assert result["iso_country_code"].unique().to_list() == ["USA"]
    assert len(result) == 2


def test_apply_filters_min_price(barra_data: pl.DataFrame):
    """Test the minimum price filter."""
    filters = [{"min_price": 15.0}]
    result = apply_filters(barra_data, filters)
    assert result["price"].min() >= 15.0
    assert len(result) == 1


def test_apply_filters_min_market_cap(barra_data: pl.DataFrame):
    """Test the minimum market cap filter."""
    filters = [{"min_market_cap": 1.5e9}]
    result = apply_filters(barra_data, filters)
    assert result["market_cap"].min() >= 1.5e9
    assert len(result) == 1


def test_apply_filters_require_returns(barra_data: pl.DataFrame):
    """Test the filter for non-null returns."""
    data_with_nulls = barra_data.with_columns(
        pl.when(pl.col("barrid") == "US123")
        .then(None)
        .otherwise(pl.col("return"))
        .alias("return")
    )
    filters = [{"require_returns": True, "usa_only": False}]
    result = apply_filters(data_with_nulls, filters) # noqa: F841
    assert result["return"].is_null().sum() == 0
    assert len(result) == 2


def test_validate_data_success(barra_data: pl.DataFrame):
    """Test that validation passes with correct data."""
    assert validate_data(barra_data) is True


def test_validate_data_failure_missing_columns():
    """Test that validation fails if a required column is missing."""
    invalid_data = pl.DataFrame({"date": [], "barrid": []})
    assert validate_data(invalid_data) is False