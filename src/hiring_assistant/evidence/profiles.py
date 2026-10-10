"""Candidate profile summaries and profile/job discovery.

Summaries are evidence-first: the deterministic path extracts labelled sections
directly from the candidate's own document (so every line is a quotation), and
the optional LLM path goes through the same citation validator as Q&A.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..models import Citation, RetrievedChunk
from ..retrieval.search import SearchService
from ..utils import collapse

#: Section labels present in the generated profile documents.
_PROFILE_SECTIONS = (
    "CORE SKILLS",
    "ADDITIONAL SKILLS",
    "SELECTED PROJECTS",
    "SUMMARY",
    "AVAILABILITY",
    "CERTIFICATIONS",
)


@dataclass
class ProfileSummary:
    """A structured, cited summary of one candidate."""

    candidate_id: str
    name: str
    headline: str = ""
    skills: list[str] = field(default_factory=list)
    additional_skills: list[str] = field(default_factory=list)
    experience_lines: list[str] = field(default_factory=list)
    project_lines: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    narrative: str = ""
    mode: str = "structured"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "headline": self.headline,
            "skills": self.skills,
            "additional_skills": self.additional_skills,
            "experience_lines": self.experience_lines,
            "project_lines": self.project_lines,
            "narrative": self.narrative,
            "mode": self.mode,
            "warnings": self.warnings,
            "citations": [c.to_dict() for c in self.citations],
        }


@dataclass
class CandidateRecord:
    """A candidate as discovered from the indexed corpus."""

    candidate_id: str
    name: str
    role_title: str
    document_id: str
    filename: str
    chunk_count: int

    def label(self) -> str:
        return f"{self.name} — {self.role_title}" if self.role_title else self.name


@dataclass
class JobRecord:
    """A job description as discovered from the indexed corpus."""

    job_id: str
    title: str
    document_id: str
    filename: str
    text: str

    def label(self) -> str:
        return self.title or self.job_id


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def discover_candidates(store: Any) -> list[CandidateRecord]:
    """Find candidate profiles among the indexed documents.

    Profiles are identified by the ``CANDIDATE PROFILE — cand_xxx`` marker in
    the generated corpus, or by living in the ``candidate_profiles``
    collection.
    """
    records: list[CandidateRecord] = []
    for document in store.list_documents():
        if document.status != "indexed":
            continue
        if Path(document.filename).name in _AGGREGATE_FILES:
            continue
        is_candidate = document.collection == "candidate_profiles" or document.filename.startswith("cand_")
        if not is_candidate:
            continue

        chunks = store.chunks_for_document(document.document_id)
        # Prefer the in-document marker; fall back to the filename, which encodes
        # the same id in the generated corpus (``cand_018_naina_kulkarni.txt``).
        candidate_id = (
            _extract_marker(chunks, r"CANDIDATE PROFILE\s*[—\-–]\s*(\S+)")
            or _id_from_filename(document.filename, r"(cand_\d+)")
            or document.document_id
        )
        name = _extract_marker(chunks, r"^Name:\s*(.+)$") or _name_from_filename(document.filename)
        role = _extract_marker(chunks, r"^Current role:\s*(.+)$") or ""

        records.append(
            CandidateRecord(
                candidate_id=candidate_id,
                name=name,
                role_title=role,
                document_id=document.document_id,
                filename=document.filename,
                chunk_count=document.chunk_count,
            )
        )

    records.sort(key=lambda r: r.candidate_id)
    return records


#: Aggregate files that live in a collection but are not a single record.
#: ``job_descriptions.json`` holds every role in one file, so treating it as a
#: job would produce an entry titled after the filename whose text is the whole
#: corpus — the per-role Markdown documents are the authoritative source.
_AGGREGATE_FILES = {"job_descriptions.json", "candidates.csv", "candidates.json"}


def discover_jobs(store: Any) -> list[JobRecord]:
    """Find job descriptions among the indexed documents."""
    records: list[JobRecord] = []
    for document in store.list_documents():
        if document.status != "indexed":
            continue
        if Path(document.filename).name in _AGGREGATE_FILES:
            continue
        is_job = document.collection == "job_descriptions" or document.filename.startswith("job_")
        if not is_job:
            continue

        chunks = store.chunks_for_document(document.document_id)
        text = "\n\n".join(c.text for c in chunks)
        job_id = (
            _extract_marker(chunks, r"JOB DESCRIPTION\s*[—\-–]\s*(\S+)")
            or _id_from_filename(document.filename, r"(job_[a-z]+_\d+)")
            or document.document_id
        )
        title = _extract_marker(chunks, r"^Title:\s*(.+)$") or _title_from_filename(document.filename)

        records.append(
            JobRecord(
                job_id=job_id,
                title=title,
                document_id=document.document_id,
                filename=document.filename,
                text=text,
            )
        )

    records.sort(key=lambda r: r.job_id)
    return records


def _extract_marker(chunks: Iterable[Any], pattern: str) -> str | None:
    regex = re.compile(pattern, re.MULTILINE)
    for chunk in chunks:
        match = regex.search(chunk.text)
        if match:
            value = collapse(match.group(1))
            if value:
                return value
    return None


def _id_from_filename(filename: str, pattern: str) -> str | None:
    """Recover a stable identifier from a filename, e.g. ``cand_018``."""
    match = re.search(pattern, Path(filename).stem)
    return match.group(1) if match else None


def _name_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"^cand_\d+_", "", stem)
    return stem.replace("_", " ").title() or filename


def _title_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"^job_\w+_\d+_", "", stem)
    return stem.replace("_", " ").title() or filename


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


class ProfileSummariser:
    """Builds cited candidate summaries."""

    def __init__(
        self,
        settings: Settings,
        search: SearchService,
        *,
        llm: Any | None = None,
    ) -> None:
        self.settings = settings
        self.search = search
        self.llm = llm

    def summarise(
        self,
        candidate: CandidateRecord,
        *,
        use_llm: bool = False,
    ) -> ProfileSummary:
        """Summarise a candidate from their own document, with citations."""
        from ..storage.database import MetadataStore  # typing only

        store: MetadataStore = self.search.store
        chunks = store.chunks_for_document(candidate.document_id)
        if not chunks:
            return ProfileSummary(
                candidate_id=candidate.candidate_id,
                name=candidate.name,
                warnings=["No indexed chunks were found for this candidate."],
            )

        summary = ProfileSummary(
            candidate_id=candidate.candidate_id,
            name=candidate.name,
        )
        summary.citations = [
            Citation(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                collection=chunk.collection,
                locator=chunk.locator,
                excerpt=chunk.excerpt(300),
                score=1.0,
                synthetic=chunk.synthetic,
            )
            for chunk in chunks
        ]

        full_text = "\n".join(chunk.text for chunk in chunks)
        summary.headline = _section(full_text, "SUMMARY") or _line(full_text, "Current role")
        summary.skills = _split_list(_section(full_text, "CORE SKILLS"))
        summary.additional_skills = _split_list(_section(full_text, "ADDITIONAL SKILLS"))
        summary.experience_lines = _experience_lines(full_text)
        summary.project_lines = _bullet_lines(_section(full_text, "SELECTED PROJECTS"))

        if use_llm and self.llm is not None and hasattr(self.llm, "summarise"):
            refined = self._llm_narrative(candidate, chunks)
            if refined:
                summary.narrative = refined
                summary.mode = "llm"

        return summary

    def _llm_narrative(self, candidate: CandidateRecord, chunks: list[Any]) -> str:
        from ..generation.citations import validate_citations
        from ..generation.prompts import SUMMARY_SYSTEM_POLICY, build_summary_message
        from ..models import RetrievedChunk

        retrieved = [
            RetrievedChunk(chunk=chunk, score=1.0, rank=index)
            for index, chunk in enumerate(chunks)
        ]
        try:
            raw = self.llm.summarise(
                SUMMARY_SYSTEM_POLICY,
                build_summary_message(candidate.name, retrieved),
            )
        except Exception:
            return ""

        report = validate_citations(raw, retrieved)
        if not report.is_grounded:
            return ""
        return report.cleaned_text

    def evidence_for_candidate(self, candidate: CandidateRecord) -> list[RetrievedChunk]:
        """All indexed chunks for one candidate, for the evidence panel."""
        chunks = self.search.store.chunks_for_document(candidate.document_id)
        return [
            RetrievedChunk(chunk=chunk, score=1.0, rank=index)
            for index, chunk in enumerate(chunks)
        ]


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------


def _section(text: str, label: str) -> str:
    pattern = re.compile(
        rf"^{re.escape(label)}\s*$(.*?)(?=^[A-Z][A-Z0-9 ()\-]{{2,}}\s*$|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _line(text: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s*(.+)$", text, re.MULTILINE)
    return collapse(match.group(1)) if match else ""


def _split_list(value: str) -> list[str]:
    if not value:
        return []
    value = value.replace("\n", "; ")
    parts = [collapse(p) for p in re.split(r"[;,]|\s{2,}", value)]
    return [p for p in parts if p and p.lower() not in {"none listed", "none"}]


def _bullet_lines(value: str) -> list[str]:
    if not value:
        return []
    lines = []
    for raw in value.split("\n"):
        cleaned = raw.strip().lstrip("-*•·").strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _experience_lines(text: str) -> list[str]:
    """Experience statements, kept verbatim so no inference is implied."""
    out: list[str] = []
    experience = _line(text, "Experience")
    if experience:
        out.append(experience)
    seniority = _line(text, "Seniority")
    if seniority:
        out.append(f"Seniority as stated: {seniority}")
    return out


__all__ = [
    "CandidateRecord",
    "JobRecord",
    "ProfileSummariser",
    "ProfileSummary",
    "discover_candidates",
    "discover_jobs",
]
