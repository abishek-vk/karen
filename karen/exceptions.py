"""Application-specific exceptions."""


class KarenError(Exception):
    """Base class for expected Karen errors."""


class ConfigurationError(KarenError):
    """Raised when configuration cannot be loaded or saved."""


class LocalAIError(KarenError):
    """Raised when the local AI service cannot complete a request."""
