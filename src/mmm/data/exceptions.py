# src/mmm/data/exceptions.py

class ColumnMappingError(Exception):
    """Base error for column mapping failures."""


class SourceColumnMissingError(ColumnMappingError):
    """Raised when one or more source columns from the mapping are missing in the input dataset."""


class TargetColumnCollisionError(ColumnMappingError):
    """Raised when two sources map to the same target or when targets collide with existing columns."""


class InvalidTargetColumnNameError(ColumnMappingError):
    """Raised when a mapping target column name does not respect MMM naming convention (v1)."""


class SourceNormalizationCollisionError(ColumnMappingError):
    """Raised when source column normalization creates duplicate column names."""
