"""Small, dependency-free helpers used across layers."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8", errors="replace"))


def content_hash(parts: Sequence[str]) -> str:
    """Stable hash over several logical fields (used for dedupe)."""
    joined = "␟".join(parts)
    return sha256_text(joined)


# --------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------

_WS_RE = re.compile(r"[ \t   ]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalise_text(text: str, *, keep_newlines: bool = True) -> str:
    """Normalise whitespace while preserving paragraph boundaries.

    Source documents come from PDFs, DOCX and CSV, all of which introduce
    different whitespace pathologies.  We collapse runs of spaces, strip
    zero-width characters and clamp blank lines — but never join paragraphs,
    because paragraph boundaries are the strongest signal the chunker has.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("​", "").replace("﻿", "")
    text = _CONTROL_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)

    if keep_newlines:
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    else:
        text = text.replace("\n", " ")
        text = _WS_RE.sub(" ", text)

    return text.strip()


def collapse(text: str) -> str:
    """Single-line, whitespace-collapsed rendering (for excerpts and tables)."""
    return " ".join(normalise_text(text, keep_newlines=False).split())


def truncate(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


# --------------------------------------------------------------------------
# Tokenisation (lexical, no model required)
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+#._/-]*")

#: Words that carry no retrieval signal in this domain.
STOPWORDS: frozenset[str] = frozenset(
    ["a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those", "of", "for", "to", "in", "on", "at", "by", "with", "from", "as", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "doing", "have", "has", "had", "having", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "too", "very", "can", "will", "just", "should", "now", "would", "could", "may", "might", "must", "about", "into", "over", "after", "before", "between", "during", "under", "above", "again", "further", "once", "here", "there", "also", "than", "its", "it's", "we're", "they're", "you're", "i'm", "don't", "doesn't", "didn't", "please", "tell", "show", "give", "find", "me", "list", "any", "some", "get", "need", "want", "looking", "use", "used", "uses", "using", "keep", "keeps", "keeping", "make", "makes", "making", "take", "takes", "taking", "come", "comes", "coming", "does", "doing", "done", "say", "says", "said", "know", "knows", "known", "think", "thinks", "want", "wants", "wanted", "like", "likely"]
)

#: Very common verbs that survive the stopword filter but carry no retrieval
#: signal.  Excluded when computing query-term coverage so that a question is
#: not judged "covered" merely because the passage contains "use" or "keep".
LOW_INFORMATION_TERMS: frozenset[str] = frozenset(
    ["use", "used", "uses", "using", "keep", "keeps", "keeping", "make", "makes", "making", "take", "takes", "taking", "give", "gives", "giving", "get", "gets", "getting", "need", "needs", "needed", "want", "wants", "wanted", "happen", "happens", "happened", "during", "between", "across", "within", "about"]
)


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Lowercase lexical tokens.  Deterministic and model-free."""
    tokens = _WORD_RE.findall((text or "").lower())
    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def lexical_overlap(query_tokens: Iterable[str], text: str) -> tuple[float, list[str]]:
    """Fraction of query terms present in ``text``, plus the matched terms.

    This is a *lexical* signal, not a probability and not a measure of truth.
    The UI labels it "term overlap" for exactly that reason.
    """
    q = {t for t in query_tokens if t}
    if not q:
        return 0.0, []
    present = token_set(text)
    matched = sorted(q & present)
    # Partial credit for prefix matches (e.g. "analytics" vs "analytical").
    for term in q - present:
        if len(term) >= 5 and any(tok.startswith(term[:5]) for tok in present):
            matched.append(term)
    return len(set(matched)) / len(q), sorted(set(matched))


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def extension_of(filename: str) -> str:
    return Path(filename).suffix.lower()


def safe_filename(filename: str) -> str:
    """Strip directory components and dangerous characters from an upload."""
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    name = name.strip(" .") or "document"
    return name[:180]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------


def dumps_compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def loads_or(default: Any, raw: str | None) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


# --------------------------------------------------------------------------
# Deterministic pseudo-randomness (used only by the synthetic corpus)
# --------------------------------------------------------------------------


class DeterministicRandom:
    """A tiny, dependency-free LCG.

    The corpus generator must be byte-for-byte reproducible across machines and
    Python versions.  ``random.Random`` is *not* guaranteed stable across major
    versions, so we ship our own.  This is for demo data only — never for
    anything security-relevant.
    """

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)

    def _next(self) -> int:
        self._state = (self._state * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        return self._state >> 16

    def random(self) -> float:
        return self._next() / float(1 << 32)

    def randint(self, low: int, high: int) -> int:
        if high <= low:
            return low
        return low + int(self.random() * (high - low + 1))

    def choice(self, items: Sequence[Any]) -> Any:
        if not items:
            raise ValueError("cannot choose from an empty sequence")
        return items[self.randint(0, len(items) - 1)]

    def sample(self, items: Sequence[Any], k: int) -> list[Any]:
        pool = list(items)
        k = max(0, min(k, len(pool)))
        out: list[Any] = []
        for _ in range(k):
            idx = self.randint(0, len(pool) - 1)
            out.append(pool.pop(idx))
        return out

    def shuffled(self, items: Sequence[Any]) -> list[Any]:
        return self.sample(items, len(items))


__all__ = [
    "STOPWORDS",
    "DeterministicRandom",
    "collapse",
    "content_hash",
    "dumps_compact",
    "ensure_dir",
    "extension_of",
    "human_size",
    "jaccard",
    "lexical_overlap",
    "loads_or",
    "normalise_text",
    "safe_filename",
    "sha256_bytes",
    "sha256_text",
    "token_set",
    "tokenize",
    "truncate",
]
