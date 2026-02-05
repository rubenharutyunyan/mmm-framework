from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

ROLE_SEPARATOR = "__"

ALLOWED_ROLES = {
    "target",
    "media",
    "control",
    "event",
    "baseline",
    "id",
}

DATE_COLUMN = "date"

_VALID_COL_RE = re.compile(r"^[a-z][a-z0-9_]*(?:__[a-z0-9_]+)*$")


@dataclass(frozen=True)
class ParsedName:
    role: str
    parts: tuple[str, ...]


def is_valid_column_name(name: str) -> bool:
    if name == DATE_COLUMN:
        return True
    return bool(_VALID_COL_RE.match(name))


def parse_column_name(name: str) -> Optional[ParsedName]:
    """
    Returns ParsedName(role, parts) or None if not matching the convention.
    Example: media__tv__spend -> role=media, parts=("tv","spend")
    """
    if name == DATE_COLUMN:
        return ParsedName(role="date", parts=())
    if not is_valid_column_name(name):
        return None
    if ROLE_SEPARATOR not in name:
        return None
    role, *parts = name.split(ROLE_SEPARATOR)
    if role not in ALLOWED_ROLES:
        return None
    if len(parts) == 0:
        return None
    return ParsedName(role=role, parts=tuple(parts))


def infer_role(name: str) -> Optional[str]:
    parsed = parse_column_name(name)
    return parsed.role if parsed else None
