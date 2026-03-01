from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mmm.data.dataset import MMMDataSet
from mmm.eda._selector import resolve_columns


@dataclass(frozen=True)
class RidgeSanityReport:
    """Result of a Ridge regression sanity check.

    Attributes
    ----------
    coefficients:
        ``{column_name: coefficient}`` for each predictor.
    intercept:
        Fitted intercept term.
    r2_score:
        In-sample R² (coefficient of determination).
    target_col:
        Target column used as the dependent variable.
    alpha:
        Ridge regularization strength used during fitting.
    """

    coefficients: dict[str, float]
    intercept: float
    r2_score: float
    target_col: str
    alpha: float


def compute_ridge_sanity(
    dataset: MMMDataSet,
    target_col: str,
    alpha: float = 1.0,
    *,
    columns: Optional[list[str]] = None,
    roles: Optional[list[str]] = None,
) -> RidgeSanityReport:
    """Fit a Ridge regression as a frequentist sanity check before Bayesian modeling.

    The input ``dataset`` is never modified; predictor and target arrays are
    extracted into transient in-memory numpy arrays.

    Parameters
    ----------
    dataset:
        A validated ``MMMDataSet``.
    target_col:
        Name of the target column. Must be explicitly provided.
        Always excluded from predictors. Must follow naming convention v1.
    alpha:
        Ridge regularization strength (default ``1.0``).
    columns:
        Explicit list of predictor columns. Takes priority over ``roles``.
    roles:
        List of column roles to filter by. Takes priority over the default.

    Returns
    -------
    RidgeSanityReport
        Frozen dataclass with coefficients, intercept, and in-sample R².

    Raises
    ------
    ValueError
        If ``target_col`` is missing or does not follow naming convention v1.
        If ``columns`` / ``roles`` validation fails or the predictor set is empty.
    """
    from sklearn.linear_model import Ridge  # import here to keep it optional at module level

    predictors = resolve_columns(
        dataset, target_col=target_col, columns=columns, roles=roles
    )

    # Extract in-memory arrays; dataset.df is never mutated.
    X = dataset.df[predictors].to_numpy(dtype=float)
    y = dataset.df[target_col].to_numpy(dtype=float)

    model = Ridge(alpha=alpha)
    model.fit(X, y)

    coefficients = {
        col: float(coef) for col, coef in zip(predictors, model.coef_)
    }

    return RidgeSanityReport(
        coefficients=coefficients,
        intercept=float(model.intercept_),
        r2_score=float(model.score(X, y)),
        target_col=target_col,
        alpha=alpha,
    )
