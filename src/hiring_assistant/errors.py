"""Structured error hierarchy.

Every failure the user can plausibly cause (bad file, missing key, stale index)
raises a subclass of :class:`HiringAssistantError` with a human-readable
``message`` and an actionable ``remedy``.  The UI renders ``remedy`` directly,
which is what turns an opaque traceback into an instruction the recruiter can
follow.
"""

from __future__ import annotations

from typing import Any


class HiringAssistantError(Exception):
    """Base class for every expected, user-facing failure."""

    #: Short label shown as the error headline.
    label = "Something went wrong"
    #: Default remediation hint shown under the headline.
    default_remedy = "Try again. If the problem persists, check the troubleshooting section of the README."

    def __init__(
        self,
        message: str,
        *,
        remedy: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy or self.default_remedy
        self.context = context or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": type(self).__name__,
            "label": self.label,
            "message": self.message,
            "remedy": self.remedy,
            "context": self.context,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.label}: {self.message}"


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


class IngestionError(HiringAssistantError):
    label = "Could not ingest document"
    default_remedy = "Check the file is a supported format, is not corrupted and is under the size limit."


class UnsupportedFileTypeError(IngestionError):
    label = "Unsupported file type"
    default_remedy = "Upload a PDF, DOCX, TXT, Markdown, CSV or JSON file."


class FileTooLargeError(IngestionError):
    label = "File too large"
    default_remedy = (
        "Reduce the file size, or raise the MAX_UPLOAD_MB limit in your .env file."
    )


class EmptyDocumentError(IngestionError):
    label = "No extractable text"
    default_remedy = (
        "The file parsed successfully but contained no text. Scanned PDFs need OCR, "
        "which this MVP does not perform — supply a text-based PDF or another format."
    )


class ParseError(IngestionError):
    label = "Could not parse document"
    default_remedy = "The file may be malformed or password-protected. Re-export it and try again."


class DuplicateDocumentError(IngestionError):
    label = "Duplicate document"
    default_remedy = "This file has the same content hash as a document already in the knowledge base."


# --------------------------------------------------------------------------
# Embeddings / providers
# --------------------------------------------------------------------------


class EmbeddingError(HiringAssistantError):
    label = "Embedding failed"
    default_remedy = "Check your embedding model configuration and network connection."


class ProviderError(HiringAssistantError):
    label = "Model provider error"
    default_remedy = "The external provider rejected the request. The app falls back to local models automatically."


class MissingCredentialsError(ProviderError):
    label = "Missing API credentials"
    default_remedy = "Add OPENAI_API_KEY to your .env file, or select a local model instead."


class ProviderTimeoutError(ProviderError):
    label = "Provider timed out"
    default_remedy = "The request exceeded REQUEST_TIMEOUT_SECONDS. Retry, or reduce the amount of context."


class ProviderRateLimitError(ProviderError):
    label = "Provider rate limit reached"
    default_remedy = "Wait a moment and retry, or switch to a local model for uninterrupted use."


class ProviderResponseError(ProviderError):
    label = "Unexpected provider response"
    default_remedy = "The provider returned data in an unexpected shape. Retry or switch to a local model."


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------


class IndexError_(HiringAssistantError):
    """Named with a trailing underscore to avoid shadowing the builtin."""

    label = "Vector index error"
    default_remedy = "Rebuild the index from the Knowledge Base view."


class IndexNotFoundError(IndexError_):
    label = "No index found"
    default_remedy = "Load the synthetic corpus or upload documents to build the first index."


class IndexMismatchError(IndexError_):
    label = "Index / model mismatch"
    default_remedy = (
        "The saved index was built with a different embedding model or chunking "
        "configuration. Rebuild the index to continue."
    )


class IndexCorruptError(IndexError_):
    label = "Index is unreadable"
    default_remedy = "Rebuild the index; the previous one was left untouched where possible."


# --------------------------------------------------------------------------
# Retrieval / generation
# --------------------------------------------------------------------------


class RetrievalError(HiringAssistantError):
    label = "Retrieval failed"
    default_remedy = "Check that at least one document is indexed, then retry the query."


class EmptyCorpusError(RetrievalError):
    label = "Knowledge base is empty"
    default_remedy = "Load the synthetic corpus from the Knowledge Base view, or upload documents."


class GenerationError(HiringAssistantError):
    label = "Could not draft an answer"
    default_remedy = "Retry the question. If it keeps failing, switch the generation model to Extractive."


class CitationValidationError(HiringAssistantError):
    label = "Citation validation failed"
    default_remedy = (
        "The model referenced a source that was not in the retrieved evidence. "
        "The unsupported citation was discarded rather than shown."
    )


# --------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------


class SafetyBlockedError(HiringAssistantError):
    label = "Request declined"
    default_remedy = (
        "This assistant retrieves and summarises evidence. Hiring decisions stay with people."
    )


class InputTooLongError(HiringAssistantError):
    label = "Input too long"
    default_remedy = "Shorten the question and try again."


__all__ = [
    "CitationValidationError",
    "DuplicateDocumentError",
    "EmbeddingError",
    "EmptyCorpusError",
    "EmptyDocumentError",
    "FileTooLargeError",
    "GenerationError",
    "HiringAssistantError",
    "IndexCorruptError",
    "IndexError_",
    "IndexMismatchError",
    "IndexNotFoundError",
    "IngestionError",
    "InputTooLongError",
    "MissingCredentialsError",
    "ParseError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "RetrievalError",
    "SafetyBlockedError",
    "UnsupportedFileTypeError",
]
