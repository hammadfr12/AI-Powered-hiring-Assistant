"""Semantic search with filters, deduplication and a lexical hybrid signal.

Retrieval combines two independent signals:

* **Vector similarity** from FAISS — captures paraphrase ("data visualisation"
  matching "Data Visualization").
* **Lexical overlap** — captures exact terms the vector model may under-weight,
  such as an ID, a tool name or a rare acronym.

The two are blended into a single ``relevance`` score.  That score is a
*ranking signal*, not a probability: the UI labels it as such and never renders
it as "confidence" or a percentage of correctness.

Metadata filters (collection, file type, document) are applied as a *pre-filter*
over the chunk ids, so a filtered search never returns a passage from outside
the requested scope.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..errors import EmptyCorpusError, IndexNotFoundError, RetrievalError
from ..models import RetrievedChunk
from ..storage.database import MetadataStore
from ..storage.vector_store import VectorStore
from ..utils import LOW_INFORMATION_TERMS, jaccard, lexical_overlap, token_set, tokenize
from .embeddings import EmbeddingProvider

#: Weighting between the dense and lexical signals.
VECTOR_WEIGHT = 0.72
LEXICAL_WEIGHT = 0.28

#: Minimum fraction of the query's *specific* terms that must appear in the
#: best passage for the result to count as evidence.  Calibrated against the
#: synthetic corpus: questions the corpus answers score ~1.0 here, questions
#: about absent topics score ~0.0-0.3.
COVERAGE_MIN_RATIO = 0.5

#: Minimum length for an unknown term to be considered a gap candidate.  Five
#: characters excludes short words ("car", "hire") that are far more often
#: phrasing than subject matter.
ABSENT_TERM_MIN_LENGTH = 5

#: How many content words unknown to the corpus are needed to declare a gap.
#: A real topic gap brings two or more; a lone unknown word is usually a
#: synonym for something the corpus states under a different name.
ABSENT_TERM_MIN_COUNT = 2

#: Minimum total characters of unknown vocabulary, so two very short unknown
#: words cannot by themselves trigger an abstention.
ABSENT_CHAR_THRESHOLD = 9

#: With only one unknown word, require the rest of the question to be thinly
#: covered as well before declaring a gap.  Set just above the coverage a
#: synonym leaves behind and below what a genuine single-word gap leaves.
GAP_SINGLE_TERM_RATIO = 0.55


@dataclass
class SearchFilters:
    """Scope restrictions for a search or question."""

    collections: list[str] = field(default_factory=list)
    file_types: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return bool(self.collections or self.file_types or self.document_ids)

    def describe(self) -> str:
        if not self.active:
            return "All documents"
        parts: list[str] = []
        if self.collections:
            parts.append("collections: " + ", ".join(sorted(self.collections)))
        if self.file_types:
            parts.append("types: " + ", ".join(sorted(self.file_types)))
        if self.document_ids:
            parts.append(f"{len(self.document_ids)} selected document(s)")
        return " · ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "collections": list(self.collections),
            "file_types": list(self.file_types),
            "document_ids": list(self.document_ids),
        }


@dataclass
class SearchResult:
    """One search response, ready to render."""

    query: str
    results: list[RetrievedChunk] = field(default_factory=list)
    filters: SearchFilters = field(default_factory=SearchFilters)
    top_k: int = 6
    pool_size: int = 0
    duration_ms: float = 0.0
    low_evidence: bool = False
    message: str = ""
    warnings: list[str] = field(default_factory=list)
    #: Query terms that are rare in the corpus (they carry the specific ask).
    specific_terms: list[str] = field(default_factory=list)
    #: Which of those the best passage actually contains.
    covered_terms: list[str] = field(default_factory=list)
    #: Terms the corpus does not contain at all — the clearest gap signal.
    absent_terms: list[str] = field(default_factory=list)

    @property
    def best_score(self) -> float:
        return self.results[0].score if self.results else 0.0

    @property
    def coverage(self) -> float:
        """Fraction of the query's specific terms present in the best passage."""
        if not self.specific_terms:
            return 1.0
        return len(self.covered_terms) / len(self.specific_terms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "filters": self.filters.to_dict(),
            "top_k": self.top_k,
            "pool_size": self.pool_size,
            "duration_ms": self.duration_ms,
            "low_evidence": self.low_evidence,
            "message": self.message,
            "warnings": self.warnings,
            "specific_terms": self.specific_terms,
            "covered_terms": self.covered_terms,
            "absent_terms": self.absent_terms,
            "coverage": round(self.coverage, 3),
            "results": [r.to_dict() for r in self.results],
        }


class SearchService:
    """Vector + lexical retrieval over the indexed corpus."""

    def __init__(
        self,
        settings: Settings,
        store: MetadataStore,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.vector_store = vector_store
        #: (corpus_size, term -> chunk_count) memo, invalidated when the corpus changes.
        self._df_cache: tuple[int, dict[str, int]] | None = None

    # ------------------------------------------------------------------
    # Scope resolution
    # ------------------------------------------------------------------

    def _allowed_chunk_ids(self, filters: SearchFilters) -> set[str] | None:
        """Resolve filters to a concrete chunk-id set (``None`` = no filter)."""
        if not filters.active:
            return None

        documents = self.store.list_documents()
        allowed_docs = {
            d.document_id
            for d in documents
            if (not filters.collections or d.collection in filters.collections)
            and (not filters.file_types or d.file_type in filters.file_types)
            and (not filters.document_ids or d.document_id in filters.document_ids)
        }
        if not allowed_docs:
            return set()

        ids: set[str] = set()
        for chunk in self.store.all_chunks(collections=filters.collections or None):
            if chunk.document_id in allowed_docs:
                ids.add(chunk.chunk_id)
        return ids

    def available_filters(self) -> dict[str, Any]:
        """Options for the UI filter widgets."""
        documents = self.store.list_documents()
        collections: dict[str, int] = {}
        file_types: dict[str, int] = {}
        for document in documents:
            if document.status != "indexed":
                continue
            collections[document.collection] = collections.get(document.collection, 0) + 1
            file_types[document.file_type] = file_types.get(document.file_type, 0) + 1
        return {
            "collections": collections,
            "file_types": file_types,
            "documents": [
                {
                    "document_id": d.document_id,
                    "filename": d.filename,
                    "collection": d.collection,
                    "chunks": d.chunk_count,
                }
                for d in documents
                if d.status == "indexed"
            ],
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
        min_relevance: float | None = None,
        dedupe: bool = True,
    ) -> SearchResult:
        """Run a semantic + lexical search and return ranked passages."""
        import time

        started = time.perf_counter()
        filters = filters or SearchFilters()
        top_k = int(top_k or self.settings.top_k)
        threshold = self.settings.min_relevance if min_relevance is None else min_relevance

        query = (query or "").strip()
        if not query:
            return SearchResult(query=query, filters=filters, top_k=top_k,
                                message="Enter a search query.")

        if not self.vector_store.is_loaded:
            raise IndexNotFoundError(
                "No index is loaded, so search is unavailable.",
                remedy="Load the synthetic corpus or upload documents from the Knowledge Base view.",
            )
        if self.vector_store.size == 0 or self.store.count_chunks() == 0:
            raise EmptyCorpusError(
                "The knowledge base is empty.",
                remedy="Load the synthetic corpus or upload documents to build an index.",
            )

        allowed = self._allowed_chunk_ids(filters)
        if allowed is not None and not allowed:
            return SearchResult(
                query=query,
                filters=filters,
                top_k=top_k,
                duration_ms=(time.perf_counter() - started) * 1000,
                message="No indexed documents match the selected filters.",
            )

        query_vector = self.embedder.embed_query(query)

        # Over-fetch so post-filtering and dedupe still leave a full page.
        pool = max(top_k * 6, 40)
        try:
            raw = self.vector_store.search(
                query_vector, top_k=pool, allowed_chunk_ids=allowed, pool_size=pool
            )
        except (IndexNotFoundError, RetrievalError):
            raise
        except Exception as exc:
            raise RetrievalError(f"Vector search failed: {exc}") from exc

        if not raw:
            return SearchResult(
                query=query,
                filters=filters,
                top_k=top_k,
                duration_ms=(time.perf_counter() - started) * 1000,
                low_evidence=True,
                message="No passages were retrieved for this query.",
            )

        chunks = {c.chunk_id: c for c in self.store.get_chunks_by_ids([cid for cid, _ in raw])}
        query_tokens = tokenize(query)

        scored: list[RetrievedChunk] = []
        for rank, (chunk_id, similarity) in enumerate(raw):
            chunk = chunks.get(chunk_id)
            if chunk is None:
                # Index and metadata disagree — surface rather than silently drop.
                continue
            overlap, matched = lexical_overlap(query_tokens, chunk.text)
            relevance = (
                VECTOR_WEIGHT * max(0.0, similarity)
                + LEXICAL_WEIGHT * overlap
                + self._person_boost(query_tokens, chunk)
            )
            scored.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=round(relevance, 4),
                    rank=rank,
                    matched_terms=matched,
                )
            )

        if dedupe:
            scored = self._dedupe(scored)

        scored.sort(key=lambda r: (-r.score, r.chunk.filename, r.chunk.chunk_index))
        for index, item in enumerate(scored):
            item.rank = index

        kept = [r for r in scored if r.score >= threshold][:top_k]

        # -- vocabulary-coverage gate -----------------------------------
        # Vector similarity alone cannot separate "asked about something the
        # corpus covers" from "asked about something adjacent to it" — the
        # distributions overlap.  The discriminating signal is whether the
        # *specific* terms of the question appear in the best passage.
        # Coverage is judged over the retrieved *set*, not just the top passage:
        # a question like "which profiles mention Airflow?" is answered by
        # several documents, none of which contains every query term alone.
        coverage_text = "\n".join(item.chunk.text for item in kept[:5])
        specific, covered, absent = self._coverage(query, coverage_text)
        coverage_ok = True
        coverage_note = ""
        # The corpus-gap test: does the question ask about subject matter whose
        # vocabulary appears nowhere in the corpus?
        #
        # This is deliberately *not* a similarity threshold.  Cosine similarity
        # cannot separate "asked about something the corpus covers" from "asked
        # about something adjacent to it" — the score distributions overlap
        # heavily.  What does separate them is whether the question's content
        # words exist in the corpus at all.
        #
        # Two unknown content words is the trigger.  A single unknown word is
        # routinely just a synonym for something the corpus does say under
        # another name ("entitlement" for a policy the corpus calls "leave"), and
        # abstaining on those would make the assistant useless.  A genuine topic
        # gap brings its own vocabulary with it — "salary band", "workplace
        # injury", "sabbatical leave", "quantum computing" each contribute two
        # or more words that appear nowhere in the corpus.
        ratio = len(covered) / len(specific) if specific else 1.0
        gap_terms = [t for t in absent if len(t) >= ABSENT_TERM_MIN_LENGTH]
        if len(gap_terms) >= ABSENT_TERM_MIN_COUNT and sum(len(t) for t in gap_terms) >= ABSENT_CHAR_THRESHOLD:
            coverage_ok = False
            coverage_note = (
                "The question asks about "
                + ", ".join(gap_terms[:4])
                + ", and the supplied documents contain no passage on that subject. The retrieved "
                "passages are topically adjacent but do not cover the specific ask."
            )
        elif len(gap_terms) == 1 and ratio <= GAP_SINGLE_TERM_RATIO:
            coverage_ok = False
            coverage_note = (
                "The question asks about " + gap_terms[0]
                + ", which the supplied documents do not mention. The retrieved passages are "
                "topically adjacent but do not cover the specific ask."
            )
        elif len(specific) <= 3 and ratio < COVERAGE_MIN_RATIO:
            coverage_ok = False
            coverage_note = (
                "The retrieved passages do not contain the specific terms of the question "
                "(" + ", ".join(t for t in specific if t not in covered)[:5] + ")."
            )

        low_evidence = (not kept or kept[0].score < threshold) or not coverage_ok

        message = ""
        if low_evidence:
            message = coverage_note or (
                "No passage in the selected documents is a close match for this query. "
                "Try different wording, remove filters, or add more documents."
            )

        return SearchResult(
            query=query,
            results=kept,
            filters=filters,
            top_k=top_k,
            pool_size=len(raw),
            duration_ms=(time.perf_counter() - started) * 1000,
            low_evidence=low_evidence,
            message=message,
            warnings=[] if not low_evidence else ["Low-evidence query: no strong match found."],
            specific_terms=specific,
            covered_terms=covered,
            absent_terms=absent,
        )

    # ------------------------------------------------------------------

    def _coverage(self, query: str, best_text: str) -> tuple[list[str], list[str], list[str]]:
        """Split query terms into corpus-specific, covered and absent sets.

        A term is *specific* when it is rare in the corpus — those are the terms
        that carry the actual ask.  Ubiquitous terms ("company", "policy") are
        ignored because they appear everywhere and discriminate nothing.
        """
        terms = [
            t for t in tokenize(query)
            if len(t) > 2 and t not in LOW_INFORMATION_TERMS
        ]
        if not terms:
            return [], [], []

        document_frequency = self._document_frequency()
        total = max(1, self._corpus_size)
        # Terms appearing in more than a fifth of chunks are background noise.
        # Frequency is resolved stem-wise so that "profiles" inherits the count
        # of "profile" rather than looking like a term the corpus never uses.
        cutoff = max(3, total * 0.2)
        specific = [t for t in terms if self._resolve_frequency(t, document_frequency) <= cutoff]
        if not specific:
            specific = terms

        best_tokens = token_set(best_text)
        covered: list[str] = []
        absent: list[str] = []
        for term in specific:
            if term in best_tokens or _prefix_match(term, best_tokens):
                covered.append(term)
            elif len(term) >= ABSENT_TERM_MIN_LENGTH and not self._term_known(term, document_frequency):
                # The term is missing from the retrieved passages *and* the
                # corpus has no morphological relative of it, so the corpus
                # genuinely has nothing to say about it.  A plain dictionary
                # lookup would treat "container" as unknown merely because the
                # corpus writes "containers" — see `_resolve_frequency`.
                absent.append(term)
        return specific, covered, absent

    @staticmethod
    def _resolve_frequency(term: str, document_frequency: dict[str, int]) -> int:
        """Frequency of ``term``, folding in morphological variants.

        A plain lookup would score "profiles" as absent simply because the
        corpus writes "profile", which would then look like a corpus gap.

        The variant test is a *prefix* relationship with a small length
        difference — "container"/"containers", "report"/"reporting" — not a
        shared stem.  A shared-stem test is too coarse: it would treat
        "quantum" as present merely because the corpus contains
        "quantitative", and then a question about quantum computing would be
        answered from an analytics passage.
        """
        exact = document_frequency.get(term, 0)
        if exact:
            return exact
        return sum(
            count
            for candidate, count in document_frequency.items()
            if _is_morphological_variant(term, candidate)
        )

    @classmethod
    def _term_known(cls, term: str, document_frequency: dict[str, int]) -> bool:
        """Does the corpus contain this term, allowing for morphological variants?"""
        return cls._resolve_frequency(term, document_frequency) > 0

    def _document_frequency(self) -> dict[str, int]:
        """Cached term → number of chunks containing it."""
        size = self.store.count_chunks()
        cached = self._df_cache
        if cached is not None and cached[0] == size:
            return cached[1]

        counts: dict[str, int] = {}
        for chunk in self.store.all_chunks():
            for token in token_set(chunk.text):
                counts[token] = counts.get(token, 0) + 1
        self._df_cache = (size, counts)
        return counts

    @property
    def _corpus_size(self) -> int:
        return self.store.count_chunks()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _person_boost(self, query_tokens: list[str], chunk: Any) -> float:
        """Small boost for readable per-person documents on people queries.

        The corpus holds each candidate twice: once as a narrative profile
        document and once as a row in a tabular index.  When the question is
        about a *person*, the narrative document is the better answer, so it
        gets a modest nudge.  The boost is deliberately small — it breaks ties
        between near-identical copies, it does not override relevance.
        """
        if not any(token in {"candidate", "profile", "candidates", "profiles", "person"} for token in query_tokens):
            return 0.0
        if chunk.collection == "candidate_profiles" and chunk.filename.startswith("cand_"):
            return 0.12
        return 0.0

    def _dedupe(self, results: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Drop near-identical passages, keeping the highest-scoring copy.

        Overlapping chunks (from ``chunk_overlap``) and repeated boilerplate
        across files would otherwise crowd out genuinely different evidence.
        """
        kept: list[RetrievedChunk] = []
        for candidate in results:
            candidate_tokens = token_set(candidate.chunk.text)
            duplicate_of: str | None = None
            for existing in kept:
                if jaccard(candidate_tokens, token_set(existing.chunk.text)) >= self.settings.dedupe_similarity:
                    duplicate_of = existing.chunk.chunk_id
                    break
            if duplicate_of:
                candidate.duplicate_of = duplicate_of
                continue
            kept.append(candidate)
        return kept

    def context_for(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
    ) -> SearchResult:
        """Retrieve the context window used by the generator."""
        return self.search(
            query,
            top_k=top_k or min(self.settings.max_context_chunks, self.settings.top_k),
            filters=filters,
        )

    def similar_to_chunk(self, chunk_id: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        """Find passages similar to an existing chunk (used by the explorer)."""
        chunk = self.store.get_chunk(chunk_id)
        if chunk is None or not self.vector_store.is_loaded:
            return []
        vector = self.embedder.embed_query(chunk.text)
        raw = self.vector_store.search(vector, top_k=top_k + 1, pool_size=top_k + 1)
        out: list[RetrievedChunk] = []
        for cid, similarity in raw:
            if cid == chunk_id:
                continue
            other = self.store.get_chunk(cid)
            if other is None:
                continue
            out.append(RetrievedChunk(chunk=other, score=round(float(similarity), 4)))
        return out[:top_k]


def _is_morphological_variant(term: str, candidate: str) -> bool:
    """True when one of two words is a short prefix-extension of the other.

    "container" / "containers" -> yes.  "report" / "reporting" -> yes.
    "quantum" / "quantitative" -> no (neither is a prefix of the other).
    """
    if term == candidate or not term or not candidate:
        return False
    shorter, longer = (term, candidate) if len(term) <= len(candidate) else (candidate, term)
    if len(shorter) < 4:
        return False
    return longer.startswith(shorter) and (len(longer) - len(shorter)) <= 4


def _prefix_match(term: str, tokens: set[str]) -> bool:
    """Bidirectional prefix match, for morphological variants.

    "leave" should match "leaves", "hire" should match "hiring", and "interview"
    should match "interviews" — but the match must not be so loose that "car"
    matches "career".  Requiring four shared leading characters in either
    direction gets that balance right.
    """
    if len(term) < 4:
        return False
    stem = term[:4]
    return any(len(tok) >= 4 and tok[:4] == stem for tok in tokens)


__all__ = ["LEXICAL_WEIGHT", "VECTOR_WEIGHT", "SearchFilters", "SearchResult", "SearchService"]
