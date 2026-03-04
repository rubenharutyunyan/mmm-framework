"""Tests for the EDA v1 module (src/mmm/eda/).

Coverage:
- Nominal correlation
- Nominal VIF
- VIF perfect collinearity → VIF = inf + warning
- Predictor selection priority (columns > roles > default)
- Empty predictor set → ValueError
- Ridge sanity check basic run
- EDARunner end-to-end
- Error cases: missing target_col, invalid role, invalid column
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from mmm.data.dataset import MMMDataSet
from mmm.eda.correlation import CorrelationReport, compute_correlation
from mmm.eda.ridge import RidgeSanityReport, compute_ridge_sanity
from mmm.eda.runner import EDAReport, EDARunner
from mmm.eda.vif import VIFReport, compute_vif


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_dataset() -> MMMDataSet:
    """Standard dataset with non-collinear predictors (n=20).

    tv and radio use non-monotonic patterns so the VIF design matrices are
    full-rank — avoiding the singular-matrix error that arises when all
    predictors are linear functions of the time index.
    """
    n = 20
    tv    = [3.0, 7.0, 2.0, 9.0, 5.0, 4.0, 8.0, 1.0, 6.0, 5.0,
             7.0, 3.0, 8.0, 2.0, 6.0, 9.0, 4.0, 5.0, 7.0, 3.0]
    radio = [8.0, 1.0, 6.0, 3.0, 9.0, 5.0, 2.0, 7.0, 4.0, 8.0,
             1.0, 6.0, 3.0, 9.0, 5.0, 2.0, 7.0, 4.0, 8.0, 1.0]
    trend = [i / (n - 1) for i in range(n)]
    price = [1.02, 1.00, 1.05, 0.97, 1.03, 0.98, 1.01, 1.04, 0.99, 1.02,
             1.00, 1.05, 0.97, 1.03, 0.98, 1.01, 1.04, 0.99, 1.02, 1.00]
    target = [tv[i] * 5.0 + radio[i] * 3.0 + trend[i] * 10.0 + 50.0
              for i in range(n)]
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="W"),
            "target__sales": target,
            "media__tv__spend": tv,
            "media__radio__spend": radio,
            "baseline__trend": trend,
            "control__price_index": price,
        }
    )
    return MMMDataSet.from_dataframe(df)


# ---------------------------------------------------------------------------
# 1. Nominal correlation
# ---------------------------------------------------------------------------


def test_correlation_nominal_shape():
    dataset = _make_dataset()
    report = compute_correlation(dataset)

    # Default selection excludes date and target; includes media, baseline, control
    expected_cols = {
        "media__tv__spend",
        "media__radio__spend",
        "baseline__trend",
        "control__price_index",
    }
    assert set(report.columns) == expected_cols
    assert isinstance(report, CorrelationReport)
    assert report.matrix.shape == (4, 4)


def test_correlation_matrix_symmetric():
    dataset = _make_dataset()
    report = compute_correlation(dataset)

    m = report.matrix
    for r in report.columns:
        for c in report.columns:
            assert math.isclose(m.loc[r, c], m.loc[c, r], abs_tol=1e-10)


def test_correlation_diagonal_is_one():
    dataset = _make_dataset()
    report = compute_correlation(dataset)

    for col in report.columns:
        assert math.isclose(report.matrix.loc[col, col], 1.0, abs_tol=1e-10)


def test_correlation_values_bounded():
    dataset = _make_dataset()
    report = compute_correlation(dataset)

    assert report.matrix.min().min() >= -1.0 - 1e-10
    assert report.matrix.max().max() <= 1.0 + 1e-10


def test_correlation_no_mutation():
    dataset = _make_dataset()
    cols_before = list(dataset.df.columns)
    compute_correlation(dataset)
    assert list(dataset.df.columns) == cols_before


# ---------------------------------------------------------------------------
# 2. Nominal VIF
# ---------------------------------------------------------------------------


def test_vif_nominal_scores_finite():
    dataset = _make_dataset()
    report = compute_vif(dataset, target_col="target__sales")

    assert isinstance(report, VIFReport)
    assert report.target_col == "target__sales"
    for col, score in report.scores.items():
        assert math.isfinite(score), f"Expected finite VIF for {col}, got {score}"
    assert report.warnings == []


def test_vif_nominal_all_predictors_present():
    dataset = _make_dataset()
    report = compute_vif(dataset, target_col="target__sales")

    expected = {
        "media__tv__spend",
        "media__radio__spend",
        "baseline__trend",
        "control__price_index",
    }
    assert set(report.scores.keys()) == expected


def test_vif_scores_positive():
    dataset = _make_dataset()
    report = compute_vif(dataset, target_col="target__sales")

    for col, score in report.scores.items():
        assert score >= 1.0, f"VIF must be >= 1, got {score} for {col}"


def test_vif_no_mutation():
    dataset = _make_dataset()
    cols_before = list(dataset.df.columns)
    compute_vif(dataset, target_col="target__sales")
    assert list(dataset.df.columns) == cols_before


# ---------------------------------------------------------------------------
# 3. VIF perfect collinearity → VIF = inf + warning
# ---------------------------------------------------------------------------


def _make_collinear_dataset() -> MMMDataSet:
    """Dataset where media__digital is the exact sum of tv and radio.

    tv and radio are linearly independent from each other and from the
    intercept, so the predictor matrix is full-rank when regressing
    tv or radio on the others.  When regressing digital on [tv, radio]
    we get R² = 1 → VIF = inf.
    """
    tv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    radio = [6.0, 3.0, 1.0, 5.0, 2.0, 4.0]
    digital = [t + r for t, r in zip(tv, radio)]  # perfect linear combination
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=6, freq="W"),
            "target__sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "media__tv": tv,
            "media__radio": radio,
            "media__digital": digital,
        }
    )
    return MMMDataSet.from_dataframe(df)


def test_vif_perfect_collinearity_inf():
    dataset = _make_collinear_dataset()
    report = compute_vif(dataset, target_col="target__sales")

    assert report.scores["media__digital"] == float("inf")


def test_vif_perfect_collinearity_warning_recorded():
    dataset = _make_collinear_dataset()
    report = compute_vif(dataset, target_col="target__sales")

    assert len(report.warnings) >= 1
    assert any("media__digital" in w for w in report.warnings)


def test_vif_perfect_collinearity_all_inf():
    # When digital = tv + radio, every column in the trio is exactly predictable
    # from the other two (tv = digital − radio, radio = digital − tv).
    # All three must therefore receive VIF = inf.
    dataset = _make_collinear_dataset()
    report = compute_vif(dataset, target_col="target__sales")

    assert report.scores["media__tv"] == float("inf")
    assert report.scores["media__radio"] == float("inf")
    assert report.scores["media__digital"] == float("inf")


# ---------------------------------------------------------------------------
# 4. Predictor selection priority
# ---------------------------------------------------------------------------


def test_selection_explicit_columns_takes_priority():
    """columns > roles > default"""
    dataset = _make_dataset()
    report = compute_correlation(
        dataset,
        columns=["media__tv__spend"],
        roles=["baseline"],  # ignored because columns is set
    )
    assert report.columns == ["media__tv__spend"]
    assert report.matrix.shape == (1, 1)


def test_selection_roles_takes_priority_over_default():
    """roles > default"""
    dataset = _make_dataset()
    report = compute_correlation(dataset, roles=["media"])

    assert set(report.columns) == {"media__tv__spend", "media__radio__spend"}


def test_selection_default_excludes_target():
    dataset = _make_dataset()
    report = compute_correlation(dataset)

    assert "target__sales" not in report.columns
    assert "date" not in report.columns


def test_selection_vif_explicit_columns():
    dataset = _make_dataset()
    report = compute_vif(
        dataset,
        target_col="target__sales",
        columns=["media__tv__spend", "media__radio__spend"],
    )
    assert set(report.scores.keys()) == {"media__tv__spend", "media__radio__spend"}


def test_selection_ridge_explicit_columns():
    dataset = _make_dataset()
    report = compute_ridge_sanity(
        dataset,
        target_col="target__sales",
        columns=["media__tv__spend", "baseline__trend"],
    )
    assert set(report.coefficients.keys()) == {"media__tv__spend", "baseline__trend"}


# ---------------------------------------------------------------------------
# 5. Empty predictor set → ValueError
# ---------------------------------------------------------------------------


def test_empty_predictor_set_no_matching_default():
    """Dataset with only target + date: default roles yield no predictors."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=5, freq="D"),
            "target__sales": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    dataset = MMMDataSet.from_dataframe(df)

    with pytest.raises(ValueError, match="No predictor columns resolved"):
        compute_correlation(dataset)


def test_empty_predictor_set_role_filter_no_match():
    dataset = _make_dataset()
    with pytest.raises(ValueError, match="No predictor columns resolved"):
        compute_correlation(dataset, roles=["event"])  # no event__ columns in dataset


def test_empty_predictor_set_vif():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=5, freq="D"),
            "target__sales": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    dataset = MMMDataSet.from_dataframe(df)
    with pytest.raises(ValueError, match="No predictor columns resolved"):
        compute_vif(dataset, target_col="target__sales")


# ---------------------------------------------------------------------------
# 6. Ridge sanity check basic run
# ---------------------------------------------------------------------------


def test_ridge_nominal_returns_report():
    dataset = _make_dataset()
    report = compute_ridge_sanity(dataset, target_col="target__sales")

    assert isinstance(report, RidgeSanityReport)
    assert report.target_col == "target__sales"
    assert report.alpha == 1.0


def test_ridge_coefficients_keys_match_predictors():
    dataset = _make_dataset()
    report = compute_ridge_sanity(dataset, target_col="target__sales")

    expected = {
        "media__tv__spend",
        "media__radio__spend",
        "baseline__trend",
        "control__price_index",
    }
    assert set(report.coefficients.keys()) == expected


def test_ridge_r2_in_valid_range():
    dataset = _make_dataset()
    report = compute_ridge_sanity(dataset, target_col="target__sales")

    assert report.r2_score <= 1.0
    # In-sample R² can be negative (extreme regularization) but almost never
    # on such a simple dataset; guard for sanity.
    assert report.r2_score >= -1.0


def test_ridge_alpha_stored():
    dataset = _make_dataset()
    report = compute_ridge_sanity(dataset, target_col="target__sales", alpha=10.0)

    assert report.alpha == 10.0


def test_ridge_intercept_is_float():
    dataset = _make_dataset()
    report = compute_ridge_sanity(dataset, target_col="target__sales")

    assert isinstance(report.intercept, float)


def test_ridge_no_mutation():
    dataset = _make_dataset()
    cols_before = list(dataset.df.columns)
    compute_ridge_sanity(dataset, target_col="target__sales")
    assert list(dataset.df.columns) == cols_before


# ---------------------------------------------------------------------------
# 7. EDARunner end-to-end
# ---------------------------------------------------------------------------


def test_eda_runner_returns_report():
    dataset = _make_dataset()
    runner = EDARunner(target_col="target__sales")
    report = runner.run(dataset)

    assert isinstance(report, EDAReport)
    assert report.target_col == "target__sales"
    assert isinstance(report.correlation, CorrelationReport)
    assert isinstance(report.vif, VIFReport)
    assert isinstance(report.ridge, RidgeSanityReport)


def test_eda_runner_columns_forwarded():
    dataset = _make_dataset()
    runner = EDARunner(
        target_col="target__sales",
        columns=["media__tv__spend", "media__radio__spend"],
    )
    report = runner.run(dataset)

    assert set(report.correlation.columns) == {
        "media__tv__spend",
        "media__radio__spend",
    }
    assert set(report.vif.scores.keys()) == {
        "media__tv__spend",
        "media__radio__spend",
    }


# ---------------------------------------------------------------------------
# 8. Error cases
# ---------------------------------------------------------------------------


def test_error_target_col_not_found():
    dataset = _make_dataset()
    with pytest.raises(ValueError, match="target_col 'target__revenue' not found"):
        compute_vif(dataset, target_col="target__revenue")


def test_error_target_col_invalid_name():
    dataset = _make_dataset()
    with pytest.raises(ValueError, match="does not follow naming convention v1"):
        compute_vif(dataset, target_col="Sales")


def test_error_invalid_role():
    dataset = _make_dataset()
    with pytest.raises(ValueError, match="not an allowed role"):
        compute_correlation(dataset, roles=["unknown_role"])


def test_error_explicit_column_missing():
    dataset = _make_dataset()
    with pytest.raises(ValueError, match="not found in dataset"):
        compute_correlation(dataset, columns=["nonexistent__col"])


def test_error_vif_fewer_than_two_predictors():
    dataset = _make_dataset()
    with pytest.raises(ValueError, match="at least 2 predictor columns"):
        compute_vif(
            dataset,
            target_col="target__sales",
            columns=["media__tv__spend"],
        )
