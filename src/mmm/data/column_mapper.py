# src/mmm/data/column_mapper.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from mmm.data.exceptions import (
    InvalidTargetColumnNameError,
    SourceColumnMissingError,
    SourceNormalizationCollisionError,
    TargetColumnCollisionError,
)

# Naming convention (v1) from docs/02_NAMING_CONVENTION.md
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
    MMM naming convention (v1)

    General format:
      <role>__<entity>__<metric>__<qualifiers...>
    - Separator: '__'
    - Case: snake_case
    - Allowed chars: a-z, 0-9, _
    - Reserved column name: 'date'
    - Allowed roles: target, media, control, event, baseline, id

    Examples:
      target__sales
      media__tv__spend
      control__price_index
      event__black_friday
    """
    import re

    if not isinstance(name, str):
        return False

    n = name.strip()
    if not n or n != name:
        return False

    # Reserved
    if n == "date":
        return True

    # Must contain at least one '__' because role__...
    if "__" not in n:
        return False

    parts = n.split("__")
    # no empty segments
    if any(p == "" for p in parts):
        return False

    role = parts[0]
    if role not in _ALLOWED_ROLES:
        return False

    # snake_case token per segment
    seg_re = re.compile(r"^[a-z][a-z0-9_]*$")
    for seg in parts[1:]:
        if not seg_re.match(seg):
            return False
        if seg.startswith("_") or seg.endswith("_"):
            return False
        if "__" in seg:
            return False  # defensive

    # Require at least one segment after role (e.g. target__sales)
    if len(parts) < 2:
        return False

    return True


@dataclass(frozen=True)
class MappingReport:
    """Traceability of mapping operations."""
    original_columns: List[str]
    normalized_columns: Optional[Dict[str, str]]  # original -> normalized (only if enabled)
    applied_mapping: Dict[str, str]               # source(after norm if enabled) -> target
    renamed_columns: Dict[str, str]               # before_rename -> after_rename
    unmapped_columns: List[str]                   # unchanged (if keep_unmapped=True)
    dropped_columns: List[str]                    # dropped (if keep_unmapped=False)


class ColumnMapper:
    """
    Apply explicit column mapping from a client dataset to MMM internal naming.

    V1 scope:
    - mapping dict: {source_col: target_col}
    - optional source normalization (snake_case-ish)
    - clear errors on:
        * missing sources
        * target collisions
        * invalid target names (naming v1)
        * normalization collisions
    - mapping_report for traceability

    Note:
    - Validation of dataset contract (date uniqueness, numeric types, etc.) is done
      later by MMMDataSet.from_dataframe(...)
    """

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
        working_mapping: Dict[str, str]

        # 1) Optional normalization on source columns (+ mapping keys)
        if self._normalize:
            normed_cols = [self._normalizer(c) for c in original_cols]

            # detect collisions induced by normalization
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

            # normalize mapping keys too
            working_mapping = {self._normalizer(k): v for k, v in self._mapping.items()}
        else:
            working_mapping = dict(self._mapping)

        # 2) Validate sources exist
        missing_sources = [src for src in working_mapping.keys() if src not in working_df.columns]
        if missing_sources:
            raise SourceColumnMissingError(
                f"Missing source column(s) in dataset: {missing_sources}"
            )

        # 3) Validate target names against naming convention (v1)
        invalid_targets = [t for t in working_mapping.values() if not is_valid_mmm_column_name(t)]
        if invalid_targets:
            raise InvalidTargetColumnNameError(
                f"Invalid target column name(s) (naming v1): {invalid_targets}"
            )

        # 4) Collision: multiple sources map to same target
        targets = list(working_mapping.values())
        dup_targets = sorted({t for t in targets if targets.count(t) > 1})
        if dup_targets:
            raise TargetColumnCollisionError(
                f"Multiple source columns map to the same target: {dup_targets}"
            )

        # 5) Collision: target collides with an unmapped existing column (when keep_unmapped=True)
        unmapped = [c for c in working_df.columns if c not in working_mapping.keys()]
        if self._keep_unmapped:
            collisions2 = sorted(set(unmapped).intersection(set(targets)))
            if collisions2:
                raise TargetColumnCollisionError(
                    "Mapping target collides with an existing unmapped column: "
                    f"{collisions2}. Either map/rename that column too, or set keep_unmapped=False."
                )

        # 6) Apply rename
        renamed_columns = dict(working_mapping)  # source -> target
        out_df = working_df.rename(columns=renamed_columns)

        # 7) Drop unmapped if requested
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
