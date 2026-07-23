"""Domain-level exceptions.

Each exception carries the HTTP status it maps to when propagated
through the API. Handlers translate them to consistent JSON responses.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all application errors mapped to HTTP responses."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DatasetError(AppError):
    """Base for errors raised while handling datasets."""

    status_code = 500
    code = "dataset_error"


class InvalidFileExtensionError(DatasetError):
    """Uploaded file has an unsupported extension."""

    status_code = 400
    code = "invalid_file_extension"


class EmptyFileError(DatasetError):
    """Uploaded file has no bytes or no data rows after parsing."""

    status_code = 400
    code = "empty_file"


class FileTooLargeError(DatasetError):
    """Uploaded file exceeds the configured size limit."""

    status_code = 413
    code = "file_too_large"


class InvalidCsvError(DatasetError):
    """Uploaded file cannot be parsed as a CSV."""

    status_code = 422
    code = "invalid_csv"


class DatasetNotFoundError(DatasetError):
    """Dataset id does not match any stored file."""

    status_code = 404
    code = "dataset_not_found"


class MLError(AppError):
    """Base for errors raised by the ML pipeline."""

    status_code = 400
    code = "ml_error"


class InvalidTargetColumnError(MLError):
    """Requested target column does not exist in the dataset."""

    status_code = 400
    code = "invalid_target_column"


class InvalidProblemTypeError(MLError):
    """Problem type is not one of the supported values."""

    status_code = 400
    code = "invalid_problem_type"


class InsufficientDataError(MLError):
    """Dataset does not have enough rows or classes to train."""

    status_code = 422
    code = "insufficient_data"


class ModelNotFoundError(MLError):
    """Model id does not match any persisted model."""

    status_code = 404
    code = "model_not_found"


class ExplainabilityError(MLError):
    """Explainability computation failed."""

    status_code = 500
    code = "explainability_error"
