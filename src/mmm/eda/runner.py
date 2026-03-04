from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mmm.data.dataset import MMMDataSet
from mmm.eda.correlation import CorrelationReport, compute_correlation
from mmm.eda.ridge import RidgeSanityReport, compute_ridge_sanity
from mmm.eda.vif import VIFReport, compute_vif


@dataclass(frozen=True)
class EDAReport:
    """Aggregated result of a full EDA run.

    Attributes
    ----------
    correlation:
        Pairwise Pearson correlation report.
    vif:
        Variance Inflation Factor report.
    ridge:
        Ridge regression sanity check report.
    target_col:
        Target column used for VIF and Ridge analyses.
    """

    correlation: CorrelationReport
    vif: VIFReport
    ridge: RidgeSanityReport
    target_col: str


class EDARunner:
    """Convenience class that runs all three EDA analyses in sequence.

    All analyses share the same ``columns`` / ``roles`` predictor selection.
    ``columns`` and ``roles`` are forwarded unchanged to each analysis.

    Parameters
    ----------
    target_col:
        Name of the target column. Used for VIF and Ridge.
        Must follow naming convention v1.
    ridge_alpha:
        Ridge regularization strength (default ``1.0``).
    columns:
        Explicit list of predictor columns. Takes priority over ``roles``.
    roles:
        List of column roles to filter by. Takes priority over the default.
    """

    def __init__(
        self,
        target_col: str,
        ridge_alpha: float = 1.0,
        *,
        columns: Optional[list[str]] = None,
        roles: Optional[list[str]] = None,
    ) -> None:
        self.target_col = target_col
        self.ridge_alpha = ridge_alpha
        self.columns = columns
        self.roles = roles

    def run(self, dataset: MMMDataSet) -> EDAReport:
        """Run all three analyses on ``dataset`` and return an ``EDAReport``.

        The input ``dataset`` is never modified.

        Parameters
        ----------
        dataset:
            A validated ``MMMDataSet``.

        Returns
        -------
        EDAReport
            Frozen dataclass aggregating all three analysis results.
        """
        correlation = compute_correlation(
            dataset, columns=self.columns, roles=self.roles
        )
        vif = compute_vif(
            dataset,
            self.target_col,
            columns=self.columns,
            roles=self.roles,
        )
        ridge = compute_ridge_sanity(
            dataset,
            self.target_col,
            self.ridge_alpha,
            columns=self.columns,
            roles=self.roles,
        )
        return EDAReport(
            correlation=correlation,
            vif=vif,
            ridge=ridge,
            target_col=self.target_col,
        )
