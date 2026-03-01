from __future__ import annotations

from typing import Optional

from mmm.config.schema import ALLOWED_ROLES, DATE_COLUMN, infer_role, parse_column_name
from mmm.data.dataset import MMMDataSet

# Default predictor roles when neither `columns` nor `roles` is specified.
DEFAULT_PREDICTOR_ROLES: frozenset[str] = frozenset({"baseline", "control", "event", "media"})


def _validate_target_col(dataset: MMMDataSet, target_col: str) -> None:
    # Naming convention checked first so that an invalid name always raises the
    # same error regardless of whether the column happens to exist in the dataset.
    parsed = parse_column_name(target_col)
    if parsed is None or parsed.role not in ALLOWED_ROLES:
        raise ValueError(f"target_col '{target_col}' does not follow naming convention v1.")
    if target_col not in dataset.df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataset columns.")


def resolve_columns(
    dataset: MMMDataSet,
    *,
    target_col: Optional[str] = None,
    columns: Optional[list[str]] = None,
    roles: Optional[list[str]] = None,
) -> list[str]:
    """Resolve predictor columns following the priority defined in docs/06_EDA.md.

    Priority order:
    1. ``columns`` — use exactly those columns (must all exist in dataset)
    2. ``roles``   — select all columns whose role matches one of the given roles
    3. default     — select columns with roles: baseline, control, event, media

    In all cases:
    - ``date`` is always excluded.
    - ``target_col`` is always excluded when provided.

    Raises
    ------
    ValueError
        If ``target_col`` is not found or does not follow naming convention v1.
        If a column listed in ``columns`` is not found in the dataset.
        If a role in ``roles`` is not an allowed role.
        If the resolved predictor set is empty.
    """
    if target_col is not None:
        _validate_target_col(dataset, target_col)

    if columns is not None:
        for col in columns:
            if col not in dataset.df.columns:
                raise ValueError(
                    f"Column '{col}' listed in `columns` not found in dataset."
                )
            if col == DATE_COLUMN:
                raise ValueError(
                    f"Column '{col}' is the date column and cannot be a predictor."
                )
            if target_col is not None and col == target_col:
                raise ValueError(
                    f"Column '{col}' is the target column and cannot be a predictor."
                )
        resolved = list(columns)

    elif roles is not None:
        for role in roles:
            if role not in ALLOWED_ROLES:
                raise ValueError(
                    f"Role '{role}' is not an allowed role (see naming convention v1)."
                )
        role_set = frozenset(roles)
        resolved = [
            c
            for c in dataset.df.columns
            if c != DATE_COLUMN
            and (target_col is None or c != target_col)
            and infer_role(c) in role_set
        ]

    else:
        resolved = [
            c
            for c in dataset.df.columns
            if c != DATE_COLUMN
            and (target_col is None or c != target_col)
            and infer_role(c) in DEFAULT_PREDICTOR_ROLES
        ]

    if not resolved:
        raise ValueError(
            "No predictor columns resolved. Check `columns`, `roles`, or dataset content."
        )

    return resolved
