from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from mmm.data.dataset import MMMDataSet
from mmm.features.report import FeatureReport


class BaseTransformer(ABC):
    """Base class for all feature transformers."""

    @abstractmethod
    def fit(self, dataset: MMMDataSet) -> "BaseTransformer":
        """Fit the transformer on the dataset (if needed)."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, dataset: MMMDataSet) -> Tuple[MMMDataSet, FeatureReport]:
        """Apply the transformation and return enriched dataset + report."""
        raise NotImplementedError

    def fit_transform(self, dataset: MMMDataSet) -> Tuple[MMMDataSet, FeatureReport]:
        self.fit(dataset)
        return self.transform(dataset)
