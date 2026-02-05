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
    mapper = ColumnMapper(
        mapping={"Date": "date", "Sales": "target__sales"},
        normalize_source_columns=False,
        keep_unmapped=True,
    )

    out, report = mapper.apply(df)

    assert "date" in out.columns
    assert "target__sales" in out.columns
    assert "Other" in out.columns

    assert report.renamed_columns == {"Date": "date", "Sales": "target__sales"}
    assert "Other" in report.unmapped_columns
    assert report.dropped_columns == []


def test_missing_source_column_raises_clear_error():
    df = pd.DataFrame({"Date": ["2024-01-01"], "Other": [1]})
    mapper = ColumnMapper(mapping={"Sales": "target__sales"})

    with pytest.raises(SourceColumnMissingError) as e:
        mapper.apply(df)

    assert "Missing source column" in str(e.value)


def test_collision_two_sources_to_one_target_raises():
    df = pd.DataFrame({"A": [1], "B": [2]})
    mapper = ColumnMapper(mapping={"A": "control__x", "B": "control__x"})

    with pytest.raises(TargetColumnCollisionError) as e:
        mapper.apply(df)

    assert "map to the same target" in str(e.value)


def test_invalid_target_name_raises_role_not_allowed():
    df = pd.DataFrame({"Sales": [10]})
    # invalid role "sales"
    mapper = ColumnMapper(mapping={"Sales": "sales__total"})

    with pytest.raises(InvalidTargetColumnNameError):
        mapper.apply(df)


def test_invalid_target_name_raises_bad_characters():
    df = pd.DataFrame({"Sales": [10]})
    # invalid: uppercase + hyphen
    mapper = ColumnMapper(mapping={"Sales": "target__Sales-Total"})

    with pytest.raises(InvalidTargetColumnNameError):
        mapper.apply(df)


def test_normalization_collision_raises():
    # "Sales " and "Sales" both normalize to "sales"
    df = pd.DataFrame({"Sales ": [1], "Sales": [2]})
    mapper = ColumnMapper(
        mapping={"Sales": "target__sales"},
        normalize_source_columns=True,
    )

    with pytest.raises(SourceNormalizationCollisionError):
        mapper.apply(df)


def test_target_collides_with_unmapped_column_raises():
    df = pd.DataFrame({"A": [1], "target__sales": [9]})
    mapper = ColumnMapper(mapping={"A": "target__sales"}, keep_unmapped=True)

    with pytest.raises(TargetColumnCollisionError) as e:
        mapper.apply(df)

    assert "collides with an existing unmapped column" in str(e.value)


def test_keep_unmapped_false_drops_unmapped_columns():
    df = pd.DataFrame({"A": [1], "B": [2]})
    mapper = ColumnMapper(mapping={"A": "control__x"}, keep_unmapped=False)

    out, report = mapper.apply(df)

    assert list(out.columns) == ["control__x"]
    assert report.unmapped_columns == []
    assert report.dropped_columns == ["B"]
