"""Grounded answer generation, citation validation and abstention."""

from .citations import (
    CitationReport,
    build_source_blocks,
    strip_citation_markers,
    validate_citations,
)
from .generator import (
    ABSTENTION_TEMPLATE,
    AnswerGenerator,
    ExtractiveGenerator,
    GenerationResult,
    LLMGenerator,
)
from .prompts import (
    REQUIREMENT_SYSTEM_POLICY,
    SUMMARY_SYSTEM_POLICY,
    SYSTEM_POLICY,
    build_context_block,
    build_user_message,
    condense_history,
)

__all__ = [
    "ABSTENTION_TEMPLATE",
    "REQUIREMENT_SYSTEM_POLICY",
    "SUMMARY_SYSTEM_POLICY",
    "SYSTEM_POLICY",
    "AnswerGenerator",
    "CitationReport",
    "ExtractiveGenerator",
    "GenerationResult",
    "LLMGenerator",
    "build_context_block",
    "build_source_blocks",
    "build_user_message",
    "condense_history",
    "strip_citation_markers",
    "validate_citations",
]
