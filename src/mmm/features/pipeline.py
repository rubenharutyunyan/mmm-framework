from __future__ import annotations

from typing import Iterable

from mmm.data.dataset import MMMDataSet
from mmm.features.base import BaseTransformer
from mmm.features.report import FeatureReport


class FeaturePipeline:
    """Sequential feature engineering pipeline."""

    def __init__(self, transformers: Iterable[BaseTransformer]) -> None:
        self.transformers = list(transformers)

    def run(self, dataset: MMMDataSet) -> tuple[MMMDataSet, FeatureReport]:
        current = dataset
        report = FeatureReport()

        for transformer in self.transformers:
            current, step_report = transformer.fit_transform(current)
            report.steps.extend(step_report.steps)

        # Final validation (contract enforcement)
        current = MMMDataSet.from_dataframe(current.df)

        return current, report
