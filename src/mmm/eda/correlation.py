from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from mmm.data.dataset import MMMDataSet
from mmm.eda._selector import resolve_columns


@dataclass(frozen=True)
class CorrelationReport:
    """Result of a pairwise Pearson correlation analysis.

    Attributes
    ----------
    matrix:
        Square symmetric DataFrame of shape (n_cols, n_cols).
        Values are in [-1, 1].
    columns:
        Column names in the same order as matrix rows/columns.
    """

    matrix: pd.DataFrame
    columns: list[str]


def compute_correlation(
    dataset: MMMDataSet,
    *,
    columns: Optional[list[str]] = None,
    roles: Optional[list[str]] = None,
) -> CorrelationReport:
    """Compute pairwise Pearson correlations between selected predictor columns.

    The input ``dataset`` is never modified.

    Parameters
    ----------
    dataset:
        A validated ``MMMDataSet``.
    columns:
        Explicit list of columns to include. Takes priority over ``roles``.
    roles:
        List of column roles to filter by. Takes priority over the default.

    Returns
    -------
    CorrelationReport
        Frozen dataclass containing the correlation matrix and column list.

    Raises
    ------
    ValueError
        If no predictor columns can be resolved, or if validation of
        ``columns`` / ``roles`` fails.
    """
    predictors = resolve_columns(dataset, columns=columns, roles=roles)

    # Operate on an in-memory slice; dataset.df is never mutated.
    df_sub = dataset.df[predictors]
    matrix = df_sub.corr(method="pearson")

    return CorrelationReport(matrix=matrix, columns=predictors)
