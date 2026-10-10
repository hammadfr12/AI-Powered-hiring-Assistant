"""Citation parsing and validation.

The single most important guarantee in this app: **a citation that is rendered
has been verified to point at a passage that was actually retrieved for this
query.**  Nothing the model writes is trusted on its own.

The model is asked to cite with inline markers of the form ``[S3]`` referencing
numbered source blocks in the prompt.  After generation we:

1. Parse every marker out of the answer.
2. Reject markers that do not correspond to a retrieved source.
3. Check the quoted span, when the model supplies one, actually appears in the
   cited passage (normalised for whitespace) — this catches fabricated quotes.
4. Rewrite unknown markers to a visible ``[unsupported reference removed]``
   token so a reader can see that something was stripped rather than silently
   losing a sentence.
5. If *no* valid citation survives, the answer is downgraded to an abstention.

Returns a :class:`CitationReport` describing exactly what happened, which the
Evaluation view reports on and the tests assert against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import Citation, RetrievedChunk
from ..utils import collapse

#: Inline marker written by the model, e.g. ``[S1]``, ``[S12]``, ``[s3]``.
_MARKER_RE = re.compile(r"\[\s*S\s*(\d{1,3})\s*\]", re.IGNORECASE)

#: Quoted span the model may attach, e.g. ``[S1: "exact quote"]`` or
#: ``[S1 "exact quote"]``.  Tolerates straight and curly quotes.
_QUOTED_MARKER_RE = re.compile(
    r"\[\s*S\s*(\d{1,3})\s*[:\-–—]?\s*[\"“”']([^\"“”']{4,400})[\"“”']\s*\]",
    re.IGNORECASE,
)

REPLACEMENT_TOKEN = "[unsupported reference removed]"

#: Confidence floor below which a quoted span is treated as unverifiable.
QUOTE_MATCH_THRESHOLD = 0.72


@dataclass
class CitationReport:
    """What happened during citation validation for one answer."""

    citations: list[Citation] = field(default_factory=list)
    valid_markers: list[int] = field(default_factory=list)
    invalid_markers: list[int] = field(default_factory=list)
    unverified_quotes: list[str] = field(default_factory=list)
    cleaned_text: str = ""
    had_any_marker: bool = False
    dropped_all: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        return bool(self.citations)

    def to_dict(self) -> dict[str, object]:
        return {
            "valid_markers": self.valid_markers,
            "invalid_markers": self.invalid_markers,
            "unverified_quotes": self.unverified_quotes,
            "had_any_marker": self.had_any_marker,
            "dropped_all": self.dropped_all,
            "citation_count": len(self.citations),
            "is_grounded": self.is_grounded,
            "warnings": self.warnings,
        }


def source_label(index: int) -> str:
    """The marker the model should write for source number ``index`` (1-based)."""
    return f"[S{index}]"


def build_source_blocks(retrieved: list[RetrievedChunk], *, limit: int = 20) -> str:
    """Render retrieved passages as numbered, citable source blocks."""
    blocks: list[str] = []
    for position, item in enumerate(retrieved[:limit], start=1):
        chunk = item.chunk
        origin = "SYNTHETIC DEMO DATA" if chunk.synthetic else "USER-UPLOADED DOCUMENT"
        blocks.append(
            "\n".join(
                [
                    f"[S{position}] filename={chunk.filename} | collection={chunk.collection} "
                    f"| location={chunk.locator} | origin={origin}",
                    chunk.text.strip(),
                ]
            )
        )
    return "\n\n".join(blocks)


def validate_citations(
    answer_text: str,
    retrieved: list[RetrievedChunk],
    *,
    allow_quote_check: bool = True,
) -> CitationReport:
    """Validate every citation marker in ``answer_text``.

    ``retrieved`` is the *exact* list that was placed in the prompt, so its
    ordering defines what ``[S1]``…``[Sn]`` mean.
    """
    report = CitationReport(cleaned_text=answer_text or "")
    if not answer_text:
        return report

    by_index: dict[int, RetrievedChunk] = {
        position: item for position, item in enumerate(retrieved, start=1)
    }

    # -- quoted spans first, so we can verify them -----------------------
    verified_quotes: dict[int, str] = {}
    for match in _QUOTED_MARKER_RE.finditer(answer_text):
        index = int(match.group(1))
        quote = match.group(2).strip()
        source = by_index.get(index)
        if source is None:
            continue
        if not allow_quote_check or _quote_present(quote, source.chunk.text):
            verified_quotes[index] = quote
        else:
            report.unverified_quotes.append(quote)

    # -- all markers -----------------------------------------------------
    # Scan both forms.  A quoted marker (``[S1: "quote"]``) is not matched by
    # the bare-marker pattern, because the quote sits between the number and the
    # closing bracket — so scanning only for bare markers would silently miss
    # every reference in an answer that quotes its sources.
    markers = _MARKER_RE.findall(answer_text)
    # findall() returns tuples here because the pattern has two groups, so read
    # the index group explicitly.
    quoted_indices = [m.group(1) for m in _QUOTED_MARKER_RE.finditer(answer_text)]
    markers = list(markers) + quoted_indices
    report.had_any_marker = bool(markers)

    seen: set[int] = set()
    for raw_index in markers:
        index = int(raw_index)
        if index in by_index:
            if index not in seen:
                seen.add(index)
                report.valid_markers.append(index)
        else:
            report.invalid_markers.append(index)

    # A quoted marker whose quote could not be verified still counts as a real
    # reference to a retrieved passage: the citation itself is valid, only the
    # quotation is unverified, and the passage excerpt is shown in its place.
    for index in verified_quotes:
        if index not in seen:
            seen.add(index)
            report.valid_markers.append(index)
    for index in quoted_indices:
        numeric = int(index)
        if numeric in by_index and numeric not in seen:
            seen.add(numeric)
            report.valid_markers.append(numeric)

    # -- rewrite invalid markers ----------------------------------------
    def replace_quoted(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index in by_index:
            return f"[S{index}]"
        return REPLACEMENT_TOKEN

    def replace_bare(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return f"[S{index}]" if index in by_index else REPLACEMENT_TOKEN

    cleaned = _QUOTED_MARKER_RE.sub(replace_quoted, answer_text)
    cleaned = _MARKER_RE.sub(replace_bare, cleaned)
    # Collapse repeated replacement tokens from a run of bogus markers.
    cleaned = re.sub(
        rf"(?:{re.escape(REPLACEMENT_TOKEN)}\s*){{2,}}", REPLACEMENT_TOKEN + " ", cleaned
    )
    report.cleaned_text = cleaned.strip()

    # -- build the citation objects -------------------------------------
    scores = {item.chunk.chunk_id: item.score for item in retrieved}
    for index in sorted(seen):
        item = by_index[index]
        chunk = item.chunk
        quote = verified_quotes.get(index)
        report.citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                collection=chunk.collection,
                locator=chunk.locator,
                excerpt=quote or chunk.excerpt(360),
                score=float(scores.get(chunk.chunk_id, item.score)),
                synthetic=chunk.synthetic,
            )
        )

    if report.invalid_markers:
        report.warnings.append(
            f"Removed {len(report.invalid_markers)} reference(s) that did not match any retrieved "
            "passage: " + ", ".join(f"[S{i}]" for i in sorted(set(report.invalid_markers))[:8])
        )
    if report.unverified_quotes:
        report.warnings.append(
            f"{len(report.unverified_quotes)} quoted span(s) could not be located verbatim in the "
            "cited passage and were not shown as quotes."
        )
    if report.had_any_marker and not report.citations:
        report.dropped_all = True
        report.warnings.append(
            "Every citation in the draft was unsupported by the retrieved evidence, so the answer "
            "was withheld rather than shown with unverifiable references."
        )

    return report


def _quote_present(quote: str, passage: str, *, threshold: float = QUOTE_MATCH_THRESHOLD) -> bool:
    """Is ``quote`` (approximately) a substring of ``passage``?

    Compared on whitespace-collapsed, case-folded text.  A small amount of
    tolerance is allowed because models routinely reflow punctuation; anything
    below the threshold is treated as a fabricated quote.
    """
    if not quote or not passage:
        return False
    q = collapse(quote).lower()
    p = collapse(passage).lower()
    if len(q) < 4:
        return False
    if q in p:
        return True

    # Longest-common-substring ratio, computed with a sliding window over the
    # passage so we do not pay for a full O(n*m) table.
    window = len(q)
    best = 0
    step = max(1, window // 12)
    for start in range(0, max(1, len(p) - window + 1), step):
        candidate = p[start : start + window]
        if not candidate:
            break
        matches = sum(1 for a, b in zip(q, candidate) if a == b)
        best = max(best, matches / window)
        if best >= threshold:
            return True
    return best >= threshold


def strip_citation_markers(text: str) -> str:
    """Remove all citation markers (used for plain-text exports)."""
    return _MARKER_RE.sub("", text or "").strip()


def citation_footnote(citation: Citation, index: int) -> str:
    """One-line Markdown footnote for exports."""
    origin = "synthetic demo data" if citation.synthetic else "uploaded document"
    return (
        f"[{index}] `{citation.filename}` — {citation.locator} "
        f"({citation.collection}; {origin}; relevance {citation.score:.3f})"
    )


__all__ = [
    "REPLACEMENT_TOKEN",
    "CitationReport",
    "build_source_blocks",
    "citation_footnote",
    "source_label",
    "strip_citation_markers",
    "validate_citations",
]
