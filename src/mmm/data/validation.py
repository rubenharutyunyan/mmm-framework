from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from mmm.config.schema import DATE_COLUMN, infer_role


@dataclass
class ValidationError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def validate_dataset(df: pd.DataFrame, freq: Optional[str] = None) -> None:
    # 1) date column
    if DATE_COLUMN not in df.columns:
        raise ValidationError(f"Missing required date column '{DATE_COLUMN}'.")

    # 2) datetime + sort + unique
    s = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    if s.isna().any():
        raise ValidationError("Some values in 'date' cannot be parsed as datetime.")
    if s.duplicated().any():
        raise ValidationError("Duplicate dates detected in 'date' column.")
    if not s.is_monotonic_increasing:
        raise ValidationError("'date' must be sorted in increasing order.")

    # 3) role checks
    for col in df.columns:
        if col == DATE_COLUMN:
            continue
        role = infer_role(col)
        if role is None:
            raise ValidationError(
                f"Column '{col}' does not follow naming convention "
                f"(expected '<role>__...')."
            )

        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValidationError(f"Column '{col}' must be numeric (role={role}).")

        if role == "target" and df[col].isna().any():
            raise ValidationError(f"Target column '{col}' contains NaN.")

        if role == "media" and (df[col] < 0).any():
            raise ValidationError(f"Media column '{col}' contains negative values.")

        if role == "event":
            # allow 0/1 or [0,1]
            if ((df[col] < 0) | (df[col] > 1)).any():
                raise ValidationError(f"Event column '{col}' must be in [0, 1].")
