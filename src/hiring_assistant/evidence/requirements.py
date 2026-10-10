"""Requirement extraction and the job-requirement evidence matrix.

Two responsibilities:

1. **Extract explicit requirements** from a job description.  Only requirements
   the document actually states are used — nothing is inferred from the title
   or invented from domain knowledge.  Numbered lists and bullet lists are the
   primary signal; when neither is present we fall back to sentences beginning
   with an obligation word ("experience with…", "ability to…").

2. **Map each requirement to candidate evidence**, producing one of three
   statuses: ``supported``, ``partial`` or ``not_found``.

The output carries *no* aggregate score, no ranking and no verdict — the PRD
forbids them, and the tests assert their absence.  "Not found" is always
explained as a gap in the supplied documents, never as a statement about the
candidate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import Settings
from ..models import Citation, EvidenceMatrix, RequirementEvidence, RetrievedChunk
from ..retrieval.search import SearchFilters, SearchService
from ..utils import collapse, lexical_overlap, tokenize

# --------------------------------------------------------------------------
# Requirement extraction
# --------------------------------------------------------------------------

_NUMBERED_RE = re.compile(r"^\s*(\d{1,2})[.)]\s+(.{8,300})$")
_BULLET_RE = re.compile(r"^\s*[-*•·–—]\s+(.{8,300})$")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")

#: Section headings that introduce requirement-like content.
_REQUIREMENT_SECTIONS = (
    "requirements", "requirement", "must have", "must-have", "essential",
    "what you'll need", "what you will need", "what we're looking for",
    "what we are looking for", "qualifications", "skills", "responsibilities",
    "you should have", "you'll bring", "you will bring", "about you",
    "nice to have", "nice-to-have", "preferred", "good to have", "bonus",
)

#: Sentence openers that signal an explicit requirement outside a list.
_OBLIGATION_OPENERS = (
    "experience with", "experience in", "experience building", "experience designing",
    "experience working", "experience owning", "experience running", "experience using",
    "strong ", "solid ", "proven ", "demonstrated ", "hands-on ", "handson ",
    "ability to", "able to", "comfortable with", "familiarity with", "familiar with",
    "knowledge of", "understanding of", "working knowledge of", "expertise in",
    "skilled in", "proficiency in", "proficient in", "background in", "track record",
    "you have ", "you bring ", "you will ", "you'll ", "we need ",
)

#: Lines that are boilerplate rather than requirements.
_BOILERPLATE = (
    "apply now", "equal opportunity", "we are an equal", "about us", "about the team",
    "about this document", "fictional", "synthetic", "how to apply", "compensation",
    "benefits", "perks", "notice period", "interview process", "our values",
)

MAX_REQUIREMENTS = 14


@dataclass
class ExtractedRequirement:
    text: str
    source_line: str
    index: int
    section: str | None = None
    optional: bool = False


def extract_requirements(job_text: str, *, max_items: int = MAX_REQUIREMENTS) -> list[ExtractedRequirement]:
    """Pull explicit requirements out of a job description.

    Ordering is preserved so the matrix reads in the same order as the posting.
    """
    if not job_text or not job_text.strip():
        return []

    lines = job_text.split("\n")
    requirements: list[ExtractedRequirement] = []
    seen: set[str] = set()
    current_section: str | None = None
    in_requirement_section = False
    optional_section = False

    def add(text: str, source_line: str, optional: bool) -> None:
        cleaned = _clean_requirement(text)
        if not cleaned or len(cleaned) < 12:
            return
        key = collapse(cleaned).lower()[:90]
        if key in seen:
            return
        if any(marker in key for marker in _BOILERPLATE):
            return
        seen.add(key)
        requirements.append(
            ExtractedRequirement(
                text=cleaned,
                source_line=source_line,
                index=len(requirements),
                section=current_section,
                optional=optional,
            )
        )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            current_section = collapse(heading_match.group(1))[:80]
            lowered = current_section.lower()
            in_requirement_section = any(token in lowered for token in _REQUIREMENT_SECTIONS)
            optional_section = any(
                token in lowered
                for token in ("nice to have", "nice-to-have", "preferred", "good to have", "bonus")
            )
            continue

        # Bare uppercase heading (TXT/PDF documents).
        if stripped.isupper() and 3 < len(stripped) < 70 and not stripped.endswith("."):
            current_section = collapse(stripped)[:80]
            lowered = current_section.lower()
            in_requirement_section = any(token in lowered for token in _REQUIREMENT_SECTIONS)
            optional_section = any(
                token in lowered
                for token in ("nice to have", "nice-to-have", "preferred", "good to have", "bonus")
            )
            continue

        numbered = _NUMBERED_RE.match(stripped)
        if numbered and in_requirement_section:
            add(numbered.group(2), stripped, optional_section)
            continue

        bullet = _BULLET_RE.match(stripped)
        if bullet:
            # Bullets are treated as requirements only inside a requirement
            # section, or when they read like obligations.
            body = bullet.group(1)
            if in_requirement_section or _looks_like_requirement(body):
                add(body, stripped, optional_section)
            continue

        if in_requirement_section and _looks_like_requirement(stripped):
            add(stripped, stripped, optional_section)

    # Fallback: nothing was found inside a recognised section, so scan the whole
    # document for obligation sentences.
    if not requirements:
        for sentence in _split_sentences(job_text):
            if _looks_like_requirement(sentence):
                add(sentence, sentence, False)
            if len(requirements) >= max_items:
                break

    return requirements[:max_items]


def _looks_like_requirement(text: str) -> bool:
    lowered = collapse(text).lower()
    if len(lowered) < 12:
        return False
    return any(lowered.startswith(opener) or f" {opener}" in lowered for opener in _OBLIGATION_OPENERS)


def _clean_requirement(text: str) -> str:
    cleaned = collapse(text)
    cleaned = re.sub(r"^(?:[-*•·–—]|\d{1,2}[.)])\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;:")
    if cleaned and not cleaned.endswith("."):
        pass
    return cleaned


def _split_sentences(text: str) -> list[str]:
    from ..ingestion.chunking import split_sentences  # local import avoids a cycle

    return [s.strip() for s in split_sentences(text) if s.strip()]


# --------------------------------------------------------------------------
# Evidence mapping
# --------------------------------------------------------------------------

NOT_FOUND_RATIONALE = (
    "No passage in the supplied documents provides evidence for this requirement. "
    "This is a gap in the documents available here — it is not a statement that the "
    "candidate lacks the skill, and it should be checked with the candidate or with "
    "additional documents before drawing any conclusion."
)

SUPPORTED_RATIONALE = "A supplied passage states this requirement explicitly."

PARTIAL_RATIONALE = (
    "A supplied passage is related to this requirement but does not satisfy it in full — "
    "for example a closely related tool, or the topic mentioned without supporting detail."
)


class RequirementMatrixBuilder:
    """Builds a candidate × job requirement evidence matrix."""

    def __init__(
        self,
        settings: Settings,
        search: SearchService,
        *,
        llm: object | None = None,
    ) -> None:
        self.settings = settings
        self.search = search
        self.llm = llm  # optional LLMGenerator; unused when None

    # ------------------------------------------------------------------

    def build(
        self,
        candidate_id: str,
        candidate_name: str,
        job_id: str,
        job_title: str,
        job_text: str,
        *,
        candidate_document_ids: list[str] | None = None,
        use_llm: bool = False,
        max_requirements: int = MAX_REQUIREMENTS,
    ) -> EvidenceMatrix:
        """Map every extracted requirement against the candidate's documents."""
        requirements = extract_requirements(job_text, max_items=max_requirements)

        matrix = EvidenceMatrix(
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            job_id=job_id,
            job_title=job_title,
        )

        if not requirements:
            matrix.warnings.append(
                "No explicit requirements could be extracted from this job description. "
                "The matrix is empty rather than guessed."
            )
            return matrix

        scope_docs = candidate_document_ids or []
        filters = SearchFilters(document_ids=scope_docs) if scope_docs else SearchFilters()

        for requirement in requirements:
            row = self._assess(requirement, filters, use_llm=use_llm)
            matrix.rows.append(row)

        if any(row.status == "not_found" for row in matrix.rows):
            matrix.warnings.append(
                "Rows marked 'not found in supplied documents' mean the available documents "
                "contain no evidence for that requirement. They are a corpus gap, not a "
                "judgement about the candidate."
            )

        return matrix

    # ------------------------------------------------------------------

    def _assess(
        self,
        requirement: ExtractedRequirement,
        filters: SearchFilters,
        *,
        use_llm: bool,
    ) -> RequirementEvidence:
        try:
            result = self.search.search(
                requirement.text,
                top_k=max(3, min(self.settings.top_k, 6)),
                filters=filters,
                min_relevance=0.0,  # we apply our own status thresholds below
            )
        except Exception as exc:  # retrieval must not abort the whole matrix
            return RequirementEvidence(
                requirement=requirement.text,
                status="not_found",
                rationale=f"Retrieval failed for this requirement ({exc}).",
                best_score=0.0,
            )

        retrieved = result.results
        if not retrieved:
            return RequirementEvidence(
                requirement=requirement.text,
                status="not_found",
                rationale=NOT_FOUND_RATIONALE,
                best_score=0.0,
            )

        best = retrieved[0].score
        support_threshold = self.settings.evidence_status_supported
        partial_threshold = self.settings.evidence_status_partial

        # A requirement whose terms barely appear in the passage is not
        # evidence even if the vector score is respectable.
        requirement_tokens = tokenize(requirement.text)
        overlap, matched = lexical_overlap(requirement_tokens, retrieved[0].chunk.text)

        if best >= support_threshold and overlap >= 0.34:
            status = "supported"
            rationale = SUPPORTED_RATIONALE
        elif best >= partial_threshold or overlap >= 0.2:
            status = "partial"
            rationale = PARTIAL_RATIONALE
        else:
            return RequirementEvidence(
                requirement=requirement.text,
                status="not_found",
                rationale=NOT_FOUND_RATIONALE,
                best_score=best,
                citations=[],
            )

        evidence_chunks = [
            item for item in retrieved[:3] if item.score >= min(partial_threshold, best * 0.75)
        ] or retrieved[:1]

        citations = [
            Citation(
                chunk_id=item.chunk.chunk_id,
                document_id=item.chunk.document_id,
                filename=item.chunk.filename,
                collection=item.chunk.collection,
                locator=item.chunk.locator,
                excerpt=item.chunk.excerpt(300),
                score=item.score,
                synthetic=item.chunk.synthetic,
            )
            for item in evidence_chunks
        ]

        if matched:
            rationale = f"{rationale} Matched terms: {', '.join(matched[:6])}."

        row = RequirementEvidence(
            requirement=requirement.text,
            status=status,  # type: ignore[arg-type]
            rationale=rationale,
            citations=citations,
            best_score=best,
        )

        if use_llm and self.llm is not None and hasattr(self.llm, "classify_requirement"):
            refined = self._refine_with_llm(row, retrieved)
            if refined is not None:
                return refined
        return row

    def _refine_with_llm(
        self,
        row: RequirementEvidence,
        retrieved: list[RetrievedChunk],
    ) -> RequirementEvidence | None:
        """Optional LLM refinement.  Any failure keeps the deterministic result."""
        try:
            raw = self.llm.classify_requirement(row.requirement, retrieved)  # type: ignore[attr-defined]
        except Exception:
            return None

        if not raw:
            return None

        first_line = raw.split("\n", 1)[0].strip().upper()
        mapping = {
            "SUPPORTED": "supported",
            "PARTIAL": "partial",
            "NOT_FOUND": "not_found",
            "NOT FOUND": "not_found",
        }
        status = mapping.get(first_line.replace("-", "_"))
        if status is None:
            return None

        # Validate any citation the model added; never trust it blindly.
        from ..generation.citations import validate_citations

        report = validate_citations(raw, retrieved)
        rationale = raw.split("\n", 1)[1].strip() if "\n" in raw else row.rationale

        if status == "not_found":
            return RequirementEvidence(
                requirement=row.requirement,
                status="not_found",
                rationale=NOT_FOUND_RATIONALE,
                best_score=row.best_score,
            )

        return RequirementEvidence(
            requirement=row.requirement,
            status=status,  # type: ignore[arg-type]
            rationale=rationale or row.rationale,
            citations=report.citations or row.citations,
            best_score=row.best_score,
        )


__all__ = [
    "MAX_REQUIREMENTS",
    "NOT_FOUND_RATIONALE",
    "PARTIAL_RATIONALE",
    "SUPPORTED_RATIONALE",
    "ExtractedRequirement",
    "RequirementMatrixBuilder",
    "extract_requirements",
]
