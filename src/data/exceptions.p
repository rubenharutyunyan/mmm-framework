class ColumnMappingError(Exception):
    """Base error for column mapping failures."""


class SourceColumnMissingError(ColumnMappingError):
    pass


class TargetColumnCollisionError(ColumnMappingError):
    pass


class InvalidTargetColumnNameError(ColumnMappingError):
    pass


class SourceNormalizationCollisionError(ColumnMappingError):
    pass
