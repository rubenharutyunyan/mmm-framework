from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from mmm.data.dataset import MMMDataSet
from mmm.eda._selector import resolve_columns

# Threshold above which R² is considered "perfect collinearity".
_R2_PERFECT_COLLINEARITY = 1.0 - 1e-12


@dataclass(frozen=True)
class VIFReport:
    """Result of a Variance Inflation Factor analysis.

    Attributes
    ----------
    scores:
        ``{column_name: vif_value}`` for each predictor.
        Columns with perfect collinearity are assigned ``np.inf``.
    warnings:
        One entry per column where VIF was set to ``np.inf``.
        Empty when no perfect collinearity is detected.
    target_col:
        Target column that was excluded from predictors.
    """

    scores: dict[str, float]
    warnings: list[str]
    target_col: str


def compute_vif(
    dataset: MMMDataSet,
    target_col: str,
    *,
    columns: Optional[list[str]] = None,
    roles: Optional[list[str]] = None,
) -> VIFReport:
    """Compute Variance Inflation Factor for each predictor column.

    VIF_j = 1 / (1 - R²_j), where R²_j is the coefficient of determination
    of regressing column ``j`` on all other predictors (with intercept).

    The input ``dataset`` is never modified; all computation is performed on
    transient numpy arrays.

    Parameters
    ----------
    dataset:
        A validated ``MMMDataSet``.
    target_col:
        Name of the target column. Always excluded from predictors.
        Must follow naming convention v1.
    columns:
        Explicit list of predictor columns. Takes priority over ``roles``.
    roles:
        List of column roles to filter by. Takes priority over the default.

    Returns
    -------
    VIFReport
        Frozen dataclass with VIF scores and any perfect-collinearity warnings.

    Raises
    ------
    ValueError
        If fewer than 2 predictor columns are resolved.
        If any predictor column has zero variance.
        If the predictor matrix is singular for a given column regression.
        If ``target_col`` is missing or does not follow naming convention v1.
        If ``columns`` / ``roles`` validation fails.
    """
    predictors = resolve_columns(
        dataset, target_col=target_col, columns=columns, roles=roles
    )

    if len(predictors) < 2:
        raise ValueError("VIF requires at least 2 predictor columns.")

    # Build in-memory numpy matrix; dataset.df is never mutated.
    X = dataset.df[predictors].to_numpy(dtype=float)
    n_rows, n_cols = X.shape

    # Guard: constant columns cannot be regressors.
    for i, col in enumerate(predictors):
        if np.std(X[:, i]) == 0.0:
            raise ValueError(
                f"Column '{col}' has zero variance; VIF cannot be computed."
            )

    scores: dict[str, float] = {}
    warnings: list[str] = []

    for i, col in enumerate(predictors):
        y = X[:, i]

        # Build design matrix: intercept + all predictors except column i.
        X_rest = np.delete(X, i, axis=1)
        X_design = np.column_stack([np.ones(n_rows), X_rest])

        # Guard: singular predictor matrix (rank-deficient).
        rank = np.linalg.matrix_rank(X_design)
        if rank < X_design.shape[1]:
            raise ValueError(
                f"Predictor matrix is singular for column '{col}'; "
                "VIF cannot be computed."
            )

        # OLS regression via least squares.
        beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        y_pred = X_design @ beta

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        # ss_tot == 0 is already caught by the zero-variance guard above.
        r2 = 1.0 - ss_res / ss_tot

        if r2 >= _R2_PERFECT_COLLINEARITY:
            scores[col] = float("inf")
            warnings.append(
                f"Column '{col}' is perfectly collinear with other predictors "
                f"(R\u00b2 >= 1 - 1e-12); VIF set to inf."
            )
        else:
            scores[col] = 1.0 / (1.0 - r2)

    return VIFReport(scores=scores, warnings=warnings, target_col=target_col)
