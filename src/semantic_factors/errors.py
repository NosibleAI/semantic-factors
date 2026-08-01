"""Exception hierarchy.

Every failure this package raises derives from SemanticFactorError, so a caller can wrap a
whole factor build in one except clause without swallowing unrelated bugs.
"""
from typing import Optional


class SemanticFactorError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(SemanticFactorError):
    """A factor definition or compute option is invalid before any request is made."""


class CredentialError(SemanticFactorError):
    """A required API key is missing or was rejected."""


class WorldApiError(SemanticFactorError):
    """The NOSIBLE World API returned an unusable response."""

    def __init__(
        self,
        message: str,
        status: Optional[int] = None,
        body: Optional[str] = None
    ) -> None:
        """
        Record the HTTP status and truncated body alongside the message.

        :param message: Human-readable description of the failure.
        :param status: HTTP status code, when one was received.
        :param body: Response body, truncated by the caller.
        :return: None.
        """
        super().__init__(message)
        self.status = status
        self.body = body


class AccessDeniedError(WorldApiError):
    """The key does not cover the requested date range. Retrying will not help."""


class EmbeddingError(SemanticFactorError):
    """The embedding provider returned an unusable response."""


class NotComputedError(SemanticFactorError):
    """A result was requested before compute() ran."""


class ReturnsUnavailableError(SemanticFactorError):
    """Price history could not be obtained for the requested tickers."""
