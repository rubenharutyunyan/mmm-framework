from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from mmm.data.dataset import MMMDataSet
from mmm.features.base import BaseTransformer
from mmm.features.report import FeatureReport, FeatureStepReport


class SeasonalityTransformer(BaseTransformer):
    """Fourier-based seasonality features."""

    def __init__(
        self,
        period: int,
        order: int,
        *,
        date_col: str = "date",
    ) -> None:
        if period <= 1:
            raise ValueError("period must be > 1")
        if order < 1:
            raise ValueError("order must be >= 1")

        self.period = period
        self.order = order
        self.date_col = date_col

    def fit(self, dataset: MMMDataSet) -> "SeasonalityTransformer":
        # Stateless transformer
        return self

    def transform(self, dataset: MMMDataSet) -> tuple[MMMDataSet, FeatureReport]:
        df = dataset.df.copy()

        n = len(df)
        t = np.arange(n, dtype=float)

        added: list[str] = []

        for k in range(1, self.order + 1):
            angle = 2.0 * math.pi * k * t / self.period

            sin_col = f"baseline__seasonality__fourier__p{self.period}__k{k}__sin"
            cos_col = f"baseline__seasonality__fourier__p{self.period}__k{k}__cos"

            if sin_col in df.columns or cos_col in df.columns:
                raise ValueError(f"Seasonality column already exists for k={k}")

            df[sin_col] = np.sin(angle)
            df[cos_col] = np.cos(angle)

            added.extend([sin_col, cos_col])

        enriched = MMMDataSet.from_dataframe(df)

        report = FeatureReport()
        report.add_step(
            FeatureStepReport(
                transformer=self.__class__.__name__,
                params={
                    "period": self.period,
                    "order": self.order,
                    "date_col": self.date_col,
                },
                added_features=added,
            )
        )

        return enriched, report
