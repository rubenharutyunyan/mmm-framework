# tests/data/test_column_mapper.py

import pandas as pd
import pytest

from mmm.data.column_mapper import ColumnMapper
from mmm.data.exceptions import (
    InvalidTargetColumnNameError,
    SourceColumnMissingError,
    SourceNormalizationCollisionError,
    TargetColumnCollisionError,
)


def test_nominal_mapping_returns_report_and_renamed_df():
    df = pd.DataFrame({"Date": ["2024-01-01"], "Sales": [10], "Other": [1]})
    mapper = ColumnMapper(mapping={"Date": "date", "Sales": "target__sales"})

    out, report = mapper.apply(df)

    assert "date" in out.columns
    assert "target__sales" in out.columns
    assert "Other" in out.columns
    assert report.renamed_columns == {"Date": "date", "Sales": "target__sales"}
    assert "Other" in report.unmapped_columns


def test_missing_source_column_raises():
    df = pd.DataFrame({"Date": ["2024-01-01"]})
    mapper = ColumnMapper(mapping={"Sales": "target__sales"})
    with pytest.raises(SourceColumnMissingError):
        mapper.apply(df)


def test_collision_two_sources_to_one_target_raises():
    df = pd.DataFrame({"A": [1], "B": [2]})
    mapper = ColumnMapper(mapping={"A": "control__x", "B": "control__x"})
    with pytest.raises(TargetColumnCollisionError):
        mapper.apply(df)


def test_invalid_target_name_raises():
    df = pd.DataFrame({"Sales": [10]})
    mapper = ColumnMapper(mapping={"Sales": "sales__total"})  # invalid role
    with pytest.raises(InvalidTargetColumnNameError):
        mapper.apply(df)


def test_normalization_collision_raises():
    df = pd.DataFrame({"Sales ": [1], "Sales": [2]})
    mapper = ColumnMapper(mapping={"Sales": "target__sales"}, normalize_source_columns=True)
    with pytest.raises(SourceNormalizationCollisionError):
        mapper.apply(df)
