# src/mmm/data/column_mapper.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from .exceptions import (
    InvalidTargetColumnNameError,
    SourceColumnMissingError,
    SourceNormalizationCollisionError,
    TargetColumnCollisionError,
)

_ALLOWED_ROLES = {
    "target",
    "media",
    "control",
    "event",
    "baseline",  # optional
    "id",        # optional (future)
}


def default_normalizer(col: str) -> str:
    """
    V1 optional normalization of source (client) column names:
    - lowercase
    - trim
    - remove accents
    - replace spaces/dashes with underscores
    - keep [a-z0-9_]
    - collapse multiple underscores
    """
    import re
    import unicodedata

    s = unicodedata.normalize("NFKD", str(col))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s)
    return s


def is_valid_mmm_column_name(name: str) -> bool:
    """
    Naming convention (v1):
      <role>__<entity>__<metric>__<qualifiers...>

    - Separator: '__'
    - Case: snake_case
    - Allowed chars: a-z, 0-9, _
    - Reserved: 'date'
    - Allowed roles: target, media, control, event, baseline, id
    """
    import re

    if not isinstance(name, str):
        return False
    n = name.strip()
    if not n or n != name:
        return False

    if n == "date":
        return True

    if "__" not in n:
        return False

    parts = n.split("__")
    if any(p == "" for p in parts):
        return False

    role = parts[0]
    if role not in _ALLOWED_ROLES:
        return False

    seg_re = re.compile(r"^[a-z][a-z0-9_]*$")
    for seg in parts[1:]:
        if not seg_re.match(seg):
            return False
        if seg.startswith("_") or seg.endswith("_"):
            return False

    # must have at least role + one segment (e.g. target__sales)
    return len(parts) >= 2


@dataclass(frozen=True)
class MappingReport:
    original_columns: List[str]
    normalized_columns: Optional[Dict[str, str]]  # original -> normalized (only if enabled)
    applied_mapping: Dict[str, str]               # source(after norm if enabled) -> target
    renamed_columns: Dict[str, str]               # before_rename -> after_rename
    unmapped_columns: List[str]                   # unchanged (if keep_unmapped=True)
    dropped_columns: List[str]                    # dropped (if keep_unmapped=False)


class ColumnMapper:
    def __init__(
        self,
        mapping: Dict[str, str],
        *,
        normalize_source_columns: bool = False,
        normalizer: Optional[Callable[[str], str]] = None,
        keep_unmapped: bool = True,
    ) -> None:
        self._mapping = dict(mapping)
        self._normalize = normalize_source_columns
        self._normalizer = normalizer or default_normalizer
        self._keep_unmapped = keep_unmapped

    def apply(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, MappingReport]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        original_cols = [str(c) for c in df.columns]
        working_df = df.copy()

        normalized_map: Optional[Dict[str, str]] = None

        if self._normalize:
            normed_cols = [self._normalizer(c) for c in original_cols]

            seen: Dict[str, str] = {}
            collisions = []
            for before, after in zip(original_cols, normed_cols):
                if after in seen and seen[after] != before:
                    collisions.append((seen[after], before, after))
                else:
                    seen[after] = before

            if collisions:
                details = ", ".join([f"{a!r} & {b!r} -> {c!r}" for a, b, c in collisions])
                raise SourceNormalizationCollisionError(
                    f"Normalization produced column collisions: {details}"
                )

            normalized_map = {before: after for before, after in zip(original_cols, normed_cols)}
            working_df.columns = normed_cols
            working_mapping = {self._normalizer(k): v for k, v in self._mapping.items()}
        else:
            working_mapping = dict(self._mapping)

        missing_sources = [src for src in working_mapping.keys() if src not in working_df.columns]
        if missing_sources:
            raise SourceColumnMissingError(f"Missing source column(s) in dataset: {missing_sources}")

        invalid_targets = [t for t in working_mapping.values() if not is_valid_mmm_column_name(t)]
        if invalid_targets:
            raise InvalidTargetColumnNameError(
                f"Invalid target column name(s) (naming v1): {invalid_targets}"
            )

        targets = list(working_mapping.values())
        dup_targets = sorted({t for t in targets if targets.count(t) > 1})
        if dup_targets:
            raise TargetColumnCollisionError(
                f"Multiple source columns map to the same target: {dup_targets}"
            )

        unmapped = [c for c in working_df.columns if c not in working_mapping.keys()]
        if self._keep_unmapped:
            collisions2 = sorted(set(unmapped).intersection(set(targets)))
            if collisions2:
                raise TargetColumnCollisionError(
                    "Mapping target collides with an existing unmapped column: "
                    f"{collisions2}. Either map/rename that column too, or set keep_unmapped=False."
                )

        renamed_columns = dict(working_mapping)
        out_df = working_df.rename(columns=renamed_columns)

        dropped_columns: List[str] = []
        if not self._keep_unmapped and unmapped:
            dropped_columns = list(unmapped)
            out_df = out_df.drop(columns=unmapped)

        report = MappingReport(
            original_columns=original_cols,
            normalized_columns=normalized_map,
            applied_mapping=working_mapping,
            renamed_columns=renamed_columns,
            unmapped_columns=unmapped if self._keep_unmapped else [],
            dropped_columns=dropped_columns,
        )
        return out_df, report
