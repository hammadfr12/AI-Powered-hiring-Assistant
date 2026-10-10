"""Structure-aware chunking.

Naive fixed-width slicing cuts sentences in half and loses the page/section a
passage came from — which would make citations useless.  This chunker instead:

1. Splits each :class:`ParsedSection` into paragraphs (the strongest available
   boundary), then sentences.
2. Packs whole sentences into windows of ``chunk_size`` characters, carrying
   ``chunk_overlap`` characters of trailing context into the next window so a
   fact spanning a boundary is still retrievable.
3. Never emits an empty chunk, never emits a chunk below ``min_chunk_chars``
   (short trailing fragments are merged back into their predecessor), and
   preserves ``section``/``page`` provenance on every chunk.

Chunk ids are derived deterministically from ``(document_id, chunk_index)`` so
that re-ingesting an unchanged file produces byte-identical ids — which is what
makes the content-hash skip optimisation safe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import Chunk
from ..utils import normalise_text
from .parsers import ParsedSection

#: Sentence boundary: terminator + whitespace, tolerating common abbreviations.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])[\"')\]]*\s+"
    r"(?![a-z])"  # do not split on "e.g. foo"
)

#: Abbreviations that should not end a sentence in this domain.
_ABBREVIATIONS = (
    "e.g.", "i.e.", "etc.", "vs.", "mr.", "ms.", "mrs.", "dr.", "prof.", "inc.",
    "ltd.", "pvt.", "co.", "no.", "fig.", "approx.", "yr.", "yrs.", "mo.", "mos.",
    "b.tech", "m.tech", "b.sc.", "m.sc.", "b.e.", "m.b.a.", "ph.d.",
)

_HEADING_LIKE_RE = re.compile(r"^(?:[A-Z][A-Za-z0-9 /&,'()\-]{2,60}:?)$")


@dataclass
class ChunkingConfig:
    chunk_size: int = 900
    chunk_overlap: int = 150
    min_chunk_chars: int = 40

    def validate(self) -> None:
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


@dataclass
class ChunkingStats:
    """Reported back to the Knowledge Base view after ingestion."""

    chunks_created: int = 0
    chunks_merged: int = 0
    chunks_dropped: int = 0
    sections_processed: int = 0
    mean_chunk_chars: float = 0.0
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Splitting primitives
# --------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, protecting common abbreviations."""
    if not text:
        return []

    protected = text
    for index, abbreviation in enumerate(_ABBREVIATIONS):
        # Replace the period with a placeholder so the splitter ignores it.
        protected = re.sub(
            re.escape(abbreviation),
            abbreviation.replace(".", "․"),
            protected,
            flags=re.IGNORECASE,
        )
    # Decimal numbers (3.5) should not split either.
    protected = re.sub(r"(\d)\.(\d)", r"\1․\2", protected)

    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(protected) if p and p.strip()]
    return [p.replace("․", ".") for p in parts] or [text.strip()]


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines, then on single newlines that follow a bullet."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) > 1:
        return blocks
    # Single-newline documents (CSV/JSON renders, resumes): treat each line as
    # its own paragraph when lines look like list items or key/value pairs.
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1 and all(_looks_like_item(line) for line in lines):
        return lines
    return blocks or [text.strip()]


def _looks_like_item(line: str) -> bool:
    if line.startswith(("-", "*", "•", "·", "–", "—")):
        return True
    if re.match(r"^\d+[.)]\s", line):
        return True
    if ":" in line and len(line.split(":", 1)[0]) <= 40:
        return True
    return _HEADING_LIKE_RE.match(line) is not None


# --------------------------------------------------------------------------
# Core chunker
# --------------------------------------------------------------------------


def chunk_sections(
    sections: list[ParsedSection],
    *,
    document_id: str,
    filename: str,
    collection: str,
    file_type: str,
    embedding_model: str = "",
    config: ChunkingConfig | None = None,
    synthetic: bool = True,
) -> tuple[list[Chunk], ChunkingStats]:
    """Turn parsed sections into retrieval-ready chunks."""
    config = config or ChunkingConfig()
    config.validate()

    stats = ChunkingStats()
    chunks: list[Chunk] = []
    cursor = 0  # running offset into the reconstructed document text

    for section in sections:
        stats.sections_processed += 1
        text = normalise_text(section.text)
        if not text.strip():
            continue

        # Prefer paragraph packing; fall back to sentence packing for a single
        # monolithic paragraph.
        units = split_paragraphs(text)
        if len(units) == 1 and len(units[0]) > config.chunk_size:
            units = split_sentences(units[0])

        for window in _pack_windows(units, config):
            body = window.strip()
            if len(body) < config.min_chunk_chars:
                stats.chunks_dropped += 1
                continue

            chunk = Chunk(
                chunk_id=f"{document_id}::c{len(chunks):04d}",
                document_id=document_id,
                chunk_index=len(chunks),
                text=body,
                filename=filename,
                collection=collection,
                file_type=file_type,
                section=section.section,
                page=section.page,
                char_start=cursor,
                char_end=cursor + len(body),
                embedding_model=embedding_model,
                synthetic=synthetic,
                token_estimate=estimate_tokens(body),
            )
            chunks.append(chunk)
            cursor += len(body) + 2  # +2 for the join separator

    # Fold an undersized trailing chunk into its predecessor rather than
    # emitting a fragment that will never retrieve well.
    if len(chunks) >= 2 and len(chunks[-1].text) < config.min_chunk_chars:
        tail = chunks.pop()
        merged = f"{chunks[-1].text}\n{tail.text}".strip()
        if len(merged) <= config.chunk_size * 2:
            chunks[-1].text = merged
            chunks[-1].char_end = chunks[-1].char_start + len(merged)
            chunks[-1].token_estimate = estimate_tokens(merged)
            stats.chunks_merged += 1
        else:
            chunks.append(tail)

    # Re-index so chunk_index is contiguous and matches list position.
    for index, chunk in enumerate(chunks):
        chunk.chunk_index = index
        chunk.chunk_id = f"{document_id}::c{index:04d}"

    stats.chunks_created = len(chunks)
    stats.mean_chunk_chars = (
        sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0.0
    )
    if not chunks:
        stats.warnings.append("No chunks were produced — the document may be too short or empty.")
    return chunks, stats


def _pack_windows(units: list[str], config: ChunkingConfig) -> list[str]:
    """Greedily pack units into windows with trailing overlap."""
    windows: list[str] = []
    buffer = ""

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue

        # A single oversized unit is hard-split on word boundaries.
        if len(unit) > config.chunk_size:
            if buffer:
                windows.append(buffer)
                buffer = ""
            windows.extend(_hard_split(unit, config.chunk_size, config.chunk_overlap))
            continue

        candidate = f"{buffer}\n{unit}" if buffer else unit
        if len(candidate) <= config.chunk_size:
            buffer = candidate
            continue

        if buffer:
            windows.append(buffer)
            buffer = _overlap_tail(buffer, config.chunk_overlap)
            candidate = f"{buffer}\n{unit}" if buffer else unit
            if len(candidate) > config.chunk_size:
                windows.append(buffer)
                buffer = unit
            else:
                buffer = candidate
        else:
            buffer = unit

    if buffer.strip():
        windows.append(buffer)
    return windows


def _overlap_tail(text: str, overlap: int) -> str:
    """Return the last ``overlap`` characters, snapped to a word boundary."""
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    if 0 < space < len(tail) - 1:
        tail = tail[space + 1 :]
    return tail.strip()


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    out: list[str] = []
    buffer: list[str] = []
    length = 0

    for word in words:
        if length + len(word) + 1 > size and buffer:
            out.append(" ".join(buffer))
            keep: list[str] = []
            kept = 0
            for previous in reversed(buffer):
                if kept + len(previous) + 1 > overlap:
                    break
                keep.insert(0, previous)
                kept += len(previous) + 1
            buffer = keep
            length = kept
        buffer.append(word)
        length += len(word) + 1

    if buffer:
        out.append(" ".join(buffer))
    return [chunk for chunk in out if chunk.strip()]


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token for English)."""
    return max(1, len(text) // 4)


def build_chunk_id(document_id: str, index: int) -> str:
    return f"{document_id}::c{index:04d}"


__all__ = [
    "ChunkingConfig",
    "ChunkingStats",
    "build_chunk_id",
    "chunk_sections",
    "estimate_tokens",
    "split_paragraphs",
    "split_sentences",
]
