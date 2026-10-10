"""Answer generation with citation enforcement and abstention.

Two interchangeable backends sit behind :class:`AnswerGenerator`:

``ExtractiveGenerator``
    No model, no credentials, no cost.  It selects the sentences from the
    retrieved passages that best match the question, quotes them verbatim with
    their source markers and states plainly that the answer is assembled from
    excerpts.  This is what makes the app fully usable offline, and it is
    honest about what it is.

``LLMGenerator``
    Calls a chat-completions endpoint with the policy in
    :mod:`hiring_assistant.generation.prompts`.  Its output goes through the
    same citation validator as the extractive path — the model gets no special
    trust.

Both paths share the same post-processing pipeline, which is where the
guarantees actually live:

* guardrail screen → retrieve → relevance gate → generate → validate citations
  → downgrade to abstention if nothing survives.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..errors import (
    GenerationError,
    MissingCredentialsError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from ..models import Answer, RetrievedChunk
from ..retrieval.search import SearchFilters, SearchService
from ..safety.guardrails import check_query
from ..utils import collapse, normalise_text, token_set, tokenize
from .citations import validate_citations
from .prompts import (
    REQUIREMENT_SYSTEM_POLICY,
    SYSTEM_POLICY,
    build_requirement_message,
    build_user_message,
    condense_history,
)

# --------------------------------------------------------------------------
# Shared constants
# --------------------------------------------------------------------------

ABSTENTION_TEMPLATE = (
    "The supplied documents do not contain enough evidence to answer this. "
    "What I searched: {scope}. Nothing retrieved scored above the relevance threshold "
    "({threshold:.2f}), so I am not going to answer from general knowledge — that would risk "
    "stating something the documents do not support."
)

PARTIAL_ABSTENTION_TEMPLATE = (
    "The supplied documents cover this only partly. Here is what they do support, with "
    "citations. Anything not covered is stated explicitly rather than filled in."
)

LOW_EVIDENCE_WARNING = (
    "Weak evidence: the best matching passage scored {score:.2f}, below the relevance "
    "threshold of {threshold:.2f}. Treat this answer as provisional and check the cited "
    "passages directly."
)

EXTRACTIVE_PREAMBLE = (
    "**Extractive answer (no language model).** These are the passages from the supplied "
    "documents that best match your question, quoted verbatim. No text was generated, so every "
    "statement below is directly checkable against its source."
)

EXTRACTIVE_NO_MATCH = (
    "**Extractive answer (no language model).** I found related passages but none of them "
    "directly answer the question. The closest matches are quoted below so you can judge them "
    "yourself."
)


@dataclass
class GenerationResult:
    """Internal wrapper so callers get warnings alongside the answer."""

    answer: Answer
    warnings: list[str]
    debug: dict[str, Any]


# --------------------------------------------------------------------------
# Extractive backend
# --------------------------------------------------------------------------


class ExtractiveGenerator:
    """Selects and quotes the most relevant sentences — no model involved."""

    name = "extractive"
    external = False

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        *,
        max_sentences: int = 6,
    ) -> str:
        if not retrieved:
            return ""

        # Content words only, and only those that are not ubiquitous: a question
        # about "annual leave entitlement" should be answered by the sentence
        # containing "entitlement", not the one containing "company".
        terms = self._content_terms(question)

        scored: list[tuple[float, int, str]] = []
        for position, item in enumerate(retrieved, start=1):
            for sentence in _sentences(item.chunk.text):
                if len(sentence) < 40:
                    continue
                weight = self._sentence_weight(sentence, terms, item.score)
                scored.append((weight, position, sentence))

        if not scored:
            return ""

        scored.sort(key=lambda s: -s[0])
        best = scored[0][0]

        # Nothing in the corpus reads like an answer to this question.
        if best < 0.2:
            lines = [EXTRACTIVE_NO_MATCH, ""]
            for position, item in enumerate(retrieved[:3], start=1):
                lines.append(
                    f"- **[S{position}]** ({item.chunk.filename} · {item.chunk.locator}): "
                    f"{item.chunk.excerpt(280)}"
                )
            return "\n".join(lines)

        chosen: list[tuple[float, int, str]] = []
        seen: set[str] = set()
        per_source: dict[int, int] = {}
        for weight, position, sentence in scored:
            # Only sentences that genuinely carry the question's terms, and
            # never a sentence markedly weaker than the best one — quoting the
            # whole context block is noise, not evidence.
            if weight < max(0.34, best * 0.75):
                break
            key = collapse(sentence).lower()[:100]
            if key in seen:
                continue
            if per_source.get(position, 0) >= 2:
                continue
            seen.add(key)
            per_source[position] = per_source.get(position, 0) + 1
            chosen.append((weight, position, sentence))
            if len(chosen) >= max_sentences:
                break

        # Prefer to show the strongest passage first, then follow source order.
        chosen.sort(key=lambda s: (s[1], -s[0]))

        lines = [EXTRACTIVE_PREAMBLE, ""]
        for _weight, position, sentence in chosen:
            lines.append(f"- {sentence.strip()} [S{position}]")
        lines.append("")
        lines.append(
            "_Every line above is a verbatim excerpt. Open a citation below to see the full "
            "passage in context._"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def _content_terms(self, question: str) -> list[str]:
        """Query terms that actually discriminate, weighted by rarity.

        Uses the corpus term-document frequency when available so that rare,
        specific words outrank common ones.
        """
        tokens = [t for t in tokenize(question) if len(t) > 2]
        if not tokens:
            return []

        frequency = getattr(self.settings, "_term_frequency", None)
        if not isinstance(frequency, dict):
            return tokens
        # Rarer terms get a higher weight; unknown terms weight highest because
        # they are the ones the corpus is most likely to be missing.
        return sorted(tokens, key=lambda t: frequency.get(t, 0))

    @staticmethod
    def _sentence_weight(sentence: str, terms: list[str], passage_score: float) -> float:
        """Score a sentence by how many query terms it contains, plus rarity.

        A sentence containing the rare terms of the question is almost always
        the sentence that answers it; that is the signal we optimise for.
        """
        if not terms:
            return max(0.0, passage_score) * 0.5

        tokens = token_set(sentence)
        hits = 0.0
        total = 0.0
        for rank, term in enumerate(terms):
            # Earlier entries in `terms` are rarer, so they carry more weight.
            importance = 1.0 + (len(terms) - rank) / (len(terms) * 2)
            total += importance
            if term in tokens:
                hits += importance
            elif len(term) >= 5 and any(tok.startswith(term[:5]) for tok in tokens):
                hits += importance * 0.7

        coverage = hits / total if total else 0.0
        # Numeric answers are strong signals for "how many / how long" questions.
        numeric_bonus = 0.12 if any(ch.isdigit() for ch in sentence) else 0.0
        return round(0.8 * coverage + 0.2 * max(0.0, passage_score) + numeric_bonus, 4)


def _sentences(text: str) -> list[str]:
    from ..ingestion.chunking import split_sentences  # local import avoids a cycle

    return [s for s in split_sentences(normalise_text(text)) if s.strip()]


# --------------------------------------------------------------------------
# OpenAI backend
# --------------------------------------------------------------------------

_CHAT_URL = "https://api.openai.com/v1/chat/completions"


class LLMGenerator:
    """Chat-completions backend.  Requires an API key."""

    name = "llm"
    external = True

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_id = settings.generation_model_id
        self._url = (settings.openai_base_url or _CHAT_URL).rstrip("/")
        if self._url.endswith("/embeddings"):
            self._url = self._url[: -len("/embeddings")] + "/chat/completions"
        elif not self._url.endswith("/chat/completions"):
            self._url = self._url + "/chat/completions"

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.settings.has_openai_key:
            raise MissingCredentialsError(
                "An OpenAI generation model is selected but OPENAI_API_KEY is not set."
            )

        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.request_timeout_seconds
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # pragma: no cover
                pass
            if exc.code == 429:
                raise ProviderRateLimitError(f"OpenAI rate limit reached. {detail}") from exc
            if exc.code in (401, 403):
                raise MissingCredentialsError(
                    f"OpenAI rejected the API key (HTTP {exc.code})."
                ) from exc
            raise ProviderResponseError(f"OpenAI returned HTTP {exc.code}. {detail}") from exc
        except urllib.error.URLError as exc:
            reason = str(getattr(exc, "reason", exc))
            if "timed out" in reason.lower():
                raise ProviderTimeoutError(f"OpenAI timed out after {self.settings.request_timeout_seconds}s.") from exc
            raise ProviderResponseError(f"Could not reach OpenAI: {reason}") from exc
        except (TimeoutError, OSError) as exc:
            raise ProviderTimeoutError(f"OpenAI request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderResponseError("OpenAI returned a non-JSON response.") from exc

    def generate(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        *,
        history_summary: str | None = None,
        filters_description: str | None = None,
        system_policy: str = SYSTEM_POLICY,
    ) -> str:
        payload = {
            "model": self.model_id,
            "temperature": 0.0,
            "max_tokens": self.settings.max_answer_tokens,
            "messages": [
                {"role": "system", "content": system_policy},
                {
                    "role": "user",
                    "content": build_user_message(
                        question,
                        retrieved,
                        history_summary=history_summary,
                        filters_description=filters_description,
                    ),
                },
            ],
        }
        body = self._post(payload)
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderResponseError("OpenAI returned no completion choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            finish = choices[0].get("finish_reason")
            raise ProviderResponseError(
                f"OpenAI returned an empty message (finish_reason={finish!r})."
            )
        return content.strip()

    def classify_requirement(self, requirement: str, retrieved: list[RetrievedChunk]) -> str:
        payload = {
            "model": self.model_id,
            "temperature": 0.0,
            "max_tokens": 220,
            "messages": [
                {"role": "system", "content": REQUIREMENT_SYSTEM_POLICY},
                {"role": "user", "content": build_requirement_message(requirement, retrieved)},
            ],
        }
        body = self._post(payload)
        choices = body.get("choices") or []
        if not choices:
            raise ProviderResponseError("OpenAI returned no choices for the requirement check.")
        return str((choices[0].get("message") or {}).get("content", "")).strip()

    def summarise(self, system_policy: str, user_message: str) -> str:
        payload = {
            "model": self.model_id,
            "temperature": 0.0,
            "max_tokens": self.settings.max_answer_tokens,
            "messages": [
                {"role": "system", "content": system_policy},
                {"role": "user", "content": user_message},
            ],
        }
        body = self._post(payload)
        choices = body.get("choices") or []
        if not choices:
            raise ProviderResponseError("OpenAI returned no choices for the summary.")
        return str((choices[0].get("message") or {}).get("content", "")).strip()


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


class AnswerGenerator:
    """Retrieve → generate → validate → abstain, for both backends."""

    def __init__(
        self,
        settings: Settings,
        search: SearchService,
    ) -> None:
        self.settings = settings
        self.search = search
        self.extractive = ExtractiveGenerator(settings)
        self.llm: LLMGenerator | None = None
        if settings.generation_provider == "openai" and settings.has_openai_key:
            self.llm = LLMGenerator(settings)

        # Give the extractive selector access to corpus term frequencies so it
        # can weight rare query terms above ubiquitous ones.  Kept on the
        # settings object (rather than a module global) so parallel sessions
        # with different corpora cannot contaminate each other.
        self._attach_term_frequency()

    def _attach_term_frequency(self) -> None:
        try:
            frequency = self.search._document_frequency()
        except Exception:
            frequency = {}
        try:
            object.__setattr__(self.settings, "_term_frequency", frequency)
        except Exception:
            pass

    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return "llm" if self.llm is not None else "extractive"

    @property
    def mode_label(self) -> str:
        if self.llm is not None:
            return f"Language model ({self.llm.model_id})"
        return "Extractive evidence mode (no language model, no API key)"

    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        *,
        filters: SearchFilters | None = None,
        history: Sequence[tuple[str, str]] | None = None,
        top_k: int | None = None,
        min_relevance: float | None = None,
        enforce_guardrails: bool = True,
    ) -> GenerationResult:
        """Answer a question, grounded in retrieved evidence or not at all."""
        import time

        started = time.perf_counter()
        filters = filters or SearchFilters()
        warnings: list[str] = []
        threshold = self.settings.min_relevance if min_relevance is None else min_relevance

        # 1. Guardrails -------------------------------------------------
        if enforce_guardrails:
            decision = check_query(question, max_chars=self.settings.max_query_chars)
            if decision.blocked or decision.redirected:
                return GenerationResult(
                    answer=Answer(
                        question=question,
                        answer_text=decision.message or "This request cannot be answered.",
                        blocked=True,
                        block_reason=decision.reason,
                        mode=self.mode,
                        model_id=self.settings.generation_model_id,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        warnings=[f"Guardrail: {decision.reason}"],
                    ),
                    warnings=[f"Guardrail triggered: {decision.reason}"],
                    debug={"guardrail": decision.to_dict()},
                )

        # 2. Retrieve ---------------------------------------------------
        search_result = self.search.context_for(
            question, top_k=top_k, filters=filters
        )
        retrieved = search_result.results

        # 3. Relevance gate ---------------------------------------------
        if not retrieved:
            return GenerationResult(
                answer=Answer(
                    question=question,
                    answer_text=ABSTENTION_TEMPLATE.format(
                        scope=filters.describe(), threshold=threshold
                    ),
                    retrieved=[],
                    abstained=True,
                    abstention_reason="no_passages_above_threshold",
                    mode=self.mode,
                    model_id=self.settings.generation_model_id,
                    latency_ms=(time.perf_counter() - started) * 1000,
                ),
                warnings=warnings,
                debug={"search": search_result.to_dict()},
            )

        best_score = retrieved[0].score
        if best_score < threshold:
            return GenerationResult(
                answer=Answer(
                    question=question,
                    answer_text=ABSTENTION_TEMPLATE.format(
                        scope=filters.describe(), threshold=threshold
                    ),
                    retrieved=retrieved,
                    abstained=True,
                    abstention_reason="below_relevance_threshold",
                    mode=self.mode,
                    model_id=self.settings.generation_model_id,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    warnings=[
                        f"Best passage scored {best_score:.3f}, below the threshold of {threshold:.3f}."
                    ],
                ),
                warnings=[f"Below relevance threshold ({best_score:.3f} < {threshold:.3f})."],
                debug={"search": search_result.to_dict()},
            )

        # The retriever also applies a *corpus-gap* test: if the question asks
        # about subject matter whose vocabulary appears nowhere in the corpus,
        # no amount of similarity should produce an answer.  Honour it here —
        # otherwise a topically-adjacent passage gets quoted as if it answered
        # the question.
        if search_result.low_evidence:
            return GenerationResult(
                answer=Answer(
                    question=question,
                    answer_text=search_result.message
                    + " "
                    + ABSTENTION_TEMPLATE.format(scope=filters.describe(), threshold=threshold),
                    retrieved=retrieved,
                    abstained=True,
                    abstention_reason="corpus_gap",
                    mode=self.mode,
                    model_id=self.settings.generation_model_id,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    warnings=[search_result.message],
                ),
                warnings=[search_result.message],
                debug={"search": search_result.to_dict()},
            )

        if best_score < threshold * 1.6:
            warnings.append(
                LOW_EVIDENCE_WARNING.format(score=best_score, threshold=threshold)
            )

        # 4. Generate ---------------------------------------------------
        history_summary = condense_history(list(history or []))
        raw_text = ""
        generation_error: str | None = None

        if self.llm is not None:
            try:
                raw_text = self.llm.generate(
                    question,
                    retrieved,
                    history_summary=history_summary or None,
                    filters_description=filters.describe() if filters.active else None,
                )
            except (ProviderTimeoutError, ProviderRateLimitError, ProviderResponseError,
                    MissingCredentialsError, GenerationError) as exc:
                generation_error = exc.message
                warnings.append(
                    f"The language model call failed ({type(exc).__name__}): {exc.message} "
                    "Falling back to extractive evidence mode for this answer."
                )
                raw_text = self.extractive.generate(question, retrieved)

        if not raw_text:
            raw_text = self.extractive.generate(question, retrieved)

        if not raw_text:
            return GenerationResult(
                answer=Answer(
                    question=question,
                    answer_text=ABSTENTION_TEMPLATE.format(
                        scope=filters.describe(), threshold=threshold
                    ),
                    retrieved=retrieved,
                    abstained=True,
                    abstention_reason="no_quotable_content",
                    mode=self.mode,
                    model_id=self.settings.generation_model_id,
                    latency_ms=(time.perf_counter() - started) * 1000,
                ),
                warnings=warnings,
                debug={"search": search_result.to_dict()},
            )

        # 5. Validate citations -----------------------------------------
        report = validate_citations(raw_text, retrieved)
        warnings.extend(report.warnings)

        if not report.is_grounded:
            # No verifiable citation survived.  Never show an unsupported
            # answer — abstain instead, but keep the passages visible.
            reason = (
                "citations_invalid" if report.had_any_marker else "no_citations_produced"
            )
            text = (
                "I drafted an answer but could not tie it to the retrieved passages, so I am "
                "withholding it rather than showing unsupported claims. The passages I searched "
                "are listed below — check them directly.\n\n"
                + ABSTENTION_TEMPLATE.format(scope=filters.describe(), threshold=threshold)
            )
            return GenerationResult(
                answer=Answer(
                    question=question,
                    answer_text=text,
                    citations=[],
                    retrieved=retrieved,
                    abstained=True,
                    abstention_reason=reason,
                    mode=self.mode,
                    model_id=self.settings.generation_model_id,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    warnings=warnings,
                    invalid_citations=[f"S{i}" for i in sorted(set(report.invalid_markers))],
                ),
                warnings=warnings,
                debug={"search": search_result.to_dict(), "citations": report.to_dict()},
            )

        # 6. Assemble ---------------------------------------------------
        answer = Answer(
            question=question,
            answer_text=report.cleaned_text,
            citations=report.citations,
            retrieved=retrieved,
            abstained=False,
            mode="llm" if (self.llm is not None and generation_error is None) else "extractive",
            model_id=(
                self.settings.generation_model_id
                if (self.llm is not None and generation_error is None)
                else "extractive-evidence-only"
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
            warnings=warnings,
            invalid_citations=[f"S{i}" for i in sorted(set(report.invalid_markers))],
        )
        return GenerationResult(
            answer=answer,
            warnings=warnings,
            debug={
                "search": search_result.to_dict(),
                "citations": report.to_dict(),
                "generation_error": generation_error,
            },
        )

    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "mode_label": self.mode_label,
            "model_id": self.settings.generation_model_id,
            "provider": self.settings.generation_provider,
            "external_calls": self.settings.external_calls_enabled,
            "openai_key_present": self.settings.has_openai_key,
        }


__all__ = [
    "ABSTENTION_TEMPLATE",
    "AnswerGenerator",
    "ExtractiveGenerator",
    "GenerationResult",
    "LLMGenerator",
]
