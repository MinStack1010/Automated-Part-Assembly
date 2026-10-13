"""Application errors that can be safely returned by the HTTP layer."""

from __future__ import annotations


class ApiError(Exception):
    """An expected, client-safe error."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class NativeBackendUnavailable(ApiError):
    def __init__(self, message: str) -> None:
        super().__init__(503, "NATIVE_BACKEND_UNAVAILABLE", message)
