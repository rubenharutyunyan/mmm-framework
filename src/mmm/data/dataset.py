from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from mmm.config.schema import DATE_COLUMN, infer_role
from mmm.data.validation import validate_dataset


@dataclass(frozen=True)
class MMMDataSet:
    df: pd.DataFrame
    freq: Optional[str] = None

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, freq: Optional[str] = None) -> "MMMDataSet":
        df = df.copy()
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
        obj = cls(df=df, freq=freq)
        obj.validate()
        return obj

    def validate(self) -> None:
        validate_dataset(self.df, freq=self.freq)

    def between(self, start: str, end: str) -> "MMMDataSet":
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        mask = (self.df[DATE_COLUMN] >= start_dt) & (self.df[DATE_COLUMN] <= end_dt)
        return MMMDataSet(df=self.df.loc[mask].copy(), freq=self.freq)

    def columns_by_role(self, role: str) -> Sequence[str]:
        cols = []
        for c in self.df.columns:
            if c == DATE_COLUMN:
                continue
            if infer_role(c) == role:
                cols.append(c)
        return cols
