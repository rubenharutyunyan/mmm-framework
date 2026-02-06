from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from mmm.data.dataset import MMMDataSet
from mmm.features.base import BaseTransformer
from mmm.features.report import FeatureReport, FeatureStepReport


def _to_datetime_index(values: Iterable[object]) -> pd.DatetimeIndex:
    # Accept strings, datetime-like, pandas Timestamps
    return pd.to_datetime(list(values), errors="raise").normalize()


def _is_valid_snake_case(name: str) -> bool:
    # Strict snake_case: lowercase letters, digits, underscores; must start with a letter.
    # No double underscores because we use __ as separator at higher level.
    if not name:
        return False
    if "__" in name:
        return False
    if not ("a" <= name[0] <= "z"):
        return False
    for ch in name:
        if ch == "_":
            continue
        if "a" <= ch <= "z":
            continue
        if "0" <= ch <= "9":
            continue
        return False
    return True


@dataclass
class _EventSpec:
    name: str
    dates: pd.DatetimeIndex


class EventTransformer(BaseTransformer):
    """Create simple binary event features from dates.

    - If `dates` is provided: create single feature `event__{default_event_name}`
    - If `events` is provided: create one feature per event key `event__{event_name}`
    """

    def __init__(
        self,
        dates: Iterable[object] | None = None,
        events: Mapping[str, Iterable[object]] | None = None,
        *,
        date_col: str = "date",
        default_event_name: str = "event",
    ) -> None:
        if (dates is None and events is None) or (dates is not None and events is not None):
            raise ValueError("Provide exactly one of `dates` or `events`.")

        if not _is_valid_snake_case(default_event_name):
            raise ValueError(f"Invalid default_event_name: {default_event_name}")

        self._dates = dates
        self._events = events
        self.date_col = date_col
        self.default_event_name = default_event_name

        self._specs: list[_EventSpec] = []

    def fit(self, dataset: MMMDataSet) -> "EventTransformer":
        specs: list[_EventSpec] = []

        if self._dates is not None:
            specs.append(
                _EventSpec(
                    name=self.default_event_name,
                    dates=_to_datetime_index(self._dates),
                )
            )
        else:
            assert self._events is not None
            for name, dts in self._events.items():
                if not _is_valid_snake_case(name):
                    raise ValueError(f"Invalid event name (must be snake_case): {name}")
                specs.append(_EventSpec(name=name, dates=_to_datetime_index(dts)))

        # Deterministic order (useful for tests & traceability)
        specs.sort(key=lambda s: s.name)
        self._specs = specs
        return self

    def transform(self, dataset: MMMDataSet) -> tuple[MMMDataSet, FeatureReport]:
        df = dataset.df.copy()

        added: list[str] = []
        for spec in self._specs:
            col = f"event__{spec.name}"
            if col in df.columns:
                raise ValueError(f"Column already exists: {col}")

            # Build binary vector aligned on dataset dates
            ds_dates = pd.to_datetime(df[self.date_col]).dt.normalize()
            is_event = ds_dates.isin(spec.dates)

            df[col] = is_event.astype(float)
            added.append(col)

        enriched = MMMDataSet.from_dataframe(df)

        report = FeatureReport()
        report.add_step(
            FeatureStepReport(
                transformer=self.__class__.__name__,
                params={
                    "date_col": self.date_col,
                    "default_event_name": self.default_event_name,
                    "events": [s.name for s in self._specs],
                },
                added_features=added,
            )
        )
        return enriched, report
