from __future__ import annotations

import pandas as pd

from mmm.dataset.dataset import MMMDataSet
from mmm.features.base import BaseTransformer
from mmm.features.report import FeatureReport, FeatureStepReport


class TrendTransformer(BaseTransformer):
    """Simple linear time trend."""

    def __init__(
        self,
        date_col: str = "date",
        normalize: bool = True,
        col_name: str = "baseline__trend",
    ) -> None:
        self.date_col = date_col
        self.normalize = normalize
        self.col_name = col_name

    def fit(self, dataset: MMMDataSet) -> "TrendTransformer":
        # Stateless transformer
        return self

    def transform(self, dataset: MMMDataSet) -> tuple[MMMDataSet, FeatureReport]:
        df = dataset.df.copy()

        if self.col_name in df.columns:
            raise ValueError(f"Column already exists: {self.col_name}")

        n = len(df)
        trend = pd.Series(range(n), index=df.index, dtype=float)

        if self.normalize and n > 1:
            trend = trend / (n - 1)

        df[self.col_name] = trend

        enriched = MMMDataSet.from_dataframe(df)

        report = FeatureReport()
        report.add_step(
            FeatureStepReport(
                transformer=self.__class__.__name__,
                params={
                    "date_col": self.date_col,
                    "normalize": self.normalize,
                    "col_name": self.col_name,
                },
                added_features=[self.col_name],
            )
        )

        return enriched, report
