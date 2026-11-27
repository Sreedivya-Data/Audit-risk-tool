"""Custom exceptions for the AI Audit and Risk Analytics Tool."""

from __future__ import annotations


class MissingColumnsError(ValueError):
    """Raised when the input file lacks required columns."""

    def __init__(self, missing_columns: list[str]) -> None:
        self.missing_columns = list(missing_columns)
        names = ", ".join(self.missing_columns)
        super().__init__(
            f"Input file is missing required column(s): {names}. "
            f"Required columns: date, time, account, amount, approver, vendor, description."
        )

    def __str__(self) -> str:  # pragma: no cover - same as args message
        return (
            f"Input file is missing required column(s): "
            f"{', '.join(self.missing_columns)}. "
            f"Required columns: date, time, account, amount, approver, vendor, description."
        )


class EmptyDatasetError(ValueError):
    """Raised when the dataset is empty after cleaning, or structurally empty on load."""

    def __init__(self, message: str = "Dataset is empty after loading and cleaning.") -> None:
        self.message = message
        super().__init__(message)
