"""Query-time guardrails and untrusted-text neutralisation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .injection import HARD_BLOCK_FAMILIES, InjectionScan, scan_for_injection

OUT_OF_SCOPE_MESSAGE = (
    "This assistant retrieves and summarises evidence from the documents you supply. "
    "It does not rank candidates, score suitability or recommend who to hire — those "
    "decisions stay with people. I can show you the evidence for each requirement, and "
    "where the supplied documents contain no evidence."
)

PROTECTED_ATTRIBUTE_MESSAGE = (
    "I can't evaluate or infer protected or sensitive characteristics — including age, "
    "gender, marital or family status, religion, caste, nationality, disability or health. "
    "I can summarise the job-relevant skills, experience and projects that the supplied "
    "documents actually state."
)

# --------------------------------------------------------------------------
# Out-of-scope: decision-making and ranking
# --------------------------------------------------------------------------

_DECISION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:should|shall|would|do)\s+(?:we|i|you)\s+(?:hire|reject|shortlist|select|drop|fire)\b", "hiring_decision"),
    (r"\b(?:hire|reject|shortlist|select|fire|terminate)\s+(?:this\s+|the\s+|him|her|them|candidate)", "hiring_decision"),
    (r"\bwho\s+should\s+(?:we|i)\s+(?:hire|pick|choose|select)\b", "hiring_decision"),
    (r"\b(?:best|top|strongest|weakest|worst)\s+(?:candidate|applicant|profile)s?\b", "ranking"),
    (r"\brank\s+(?:the\s+)?(?:candidates?|applicants?|profiles?)\b", "ranking"),
    (r"\b(?:compare|order|sort)\b[^.\n]{0,30}\bcandidates?\b[^.\n]{0,30}\b(?:best|worst|better|rank)\b", "ranking"),
    (r"\b(?:suitability|fit)\s+(?:score|rating|rank|percentage)\b", "opaque_score"),
    (r"\b(?:give|assign|compute|calculate)\b[^.\n]{0,25}\b(?:a\s+)?(?:score|rating|grade)\b[^.\n]{0,25}"
     r"\b(?:candidate|applicant|profile|them|him|her)\b", "opaque_score"),
    (r"\b(?:is|would)\s+(?:this\s+|the\s+)?(?:candidate|applicant|person)\b[^.\n]{0,30}"
     r"\b(?:a\s+)?(?:good|bad|strong|weak)\s+(?:fit|hire|candidate)\b", "hiring_decision"),
    (r"\b(?:good|bad)\s+fit\s+for\s+(?:the\s+)?(?:role|job|team|company)\b", "culture_fit"),
    (r"\bculture\s+fit\b", "culture_fit"),
)

# --------------------------------------------------------------------------
# Protected / sensitive attributes
# --------------------------------------------------------------------------

_PROTECTED_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:how\s+old|age|birth\s*date|date\s+of\s+birth|dob)\b[^.\n]{0,30}"
     r"\b(?:candidate|applicant|he|she|they|this\s+person|profile)\b", "age"),
    (r"\b(?:candidate|applicant|person|profile)s?\b[^.\n]{0,30}\b(?:how\s+old|age|birth\s*date|dob)\b", "age"),
    (r"\b(?:gender|sex|male|female|man|woman|pronouns?)\b[^.\n]{0,30}"
     r"\b(?:candidate|applicant|he|she|they|profile|person)\b", "gender"),
    (r"\b(?:marital\s+status|married|single|divorced|spouse|children|kids|pregnan\w*|family\s+status|"
     r"maternity|paternity)\b", "family_status"),
    (r"\b(?:religion|religious|caste|creed|sect|community|gotra|jati)\b", "religion_caste"),
    (r"\b(?:nationality|citizenship|visa\s+status|immigration|race|ethnicity|colour|color)\b", "nationality_race"),
    (r"\b(?:disabilit\w*|handicap\w*|health\s+condition|medical\s+condition|illness|disease|"
     r"mental\s+health|diagnos\w*|hiv|chronic\s+illness)\b", "health_disability"),
    (r"\b(?:personality|culture\s+fit|emotional\s+intelligence|eq\s+score|"
     r"temperament|character\s+traits?|psychometric)\b", "personality"),
    (r"\b(?:political|affiliation|union\s+membership|sexual\s+orientation)\b", "political_orientation"),
    (r"\b(?:socioeconomic|income\s+level|family\s+background|father'?s?\s+occupation|background\s+check)\b",
     "socioeconomic"),
)

#: Phrases that mean the user wants a *decision*, even if worded indirectly.
_SOFT_REDIRECT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:is|are)\s+(?:this|the|these)\s+(?:candidate|applicant|profile)s?\s+"
     r"(?:good|qualified|suitable|ready|strong)\s+enough\b", "implicit_decision"),
    (r"\bwould\s+you\s+(?:hire|advance|reject|recommend)\b", "implicit_decision"),
    (r"\b(?:recommend|advise)\s+(?:me\s+)?(?:a\s+)?(?:candidate|applicant|profile)s?\b", "implicit_decision"),
)

_COMPILED_DECISION = tuple((re.compile(p, re.IGNORECASE), k) for p, k in _DECISION_PATTERNS)
_COMPILED_PROTECTED = tuple((re.compile(p, re.IGNORECASE), k) for p, k in _PROTECTED_PATTERNS)
_COMPILED_SOFT = tuple((re.compile(p, re.IGNORECASE), k) for p, k in _SOFT_REDIRECT_PATTERNS)


@dataclass
class GuardrailDecision:
    """Outcome of screening one user query."""

    allowed: bool = True
    blocked: bool = False
    redirected: bool = False
    reason: str | None = None
    message: str | None = None
    categories: list[str] = field(default_factory=list)
    injection: InjectionScan | None = None
    matched_text: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "blocked": self.blocked,
            "redirected": self.redirected,
            "reason": self.reason,
            "message": self.message,
            "categories": self.categories,
            "matched_text": self.matched_text,
            "injection": self.injection.to_dict() if self.injection else None,
        }


def check_query(query: str, *, max_chars: int = 2000) -> GuardrailDecision:
    """Screen a user query before retrieval.

    Three outcomes:

    * **blocked** — an attempt to extract secrets, run tools or jailbreak.  The
      request is refused and nothing is retrieved.
    * **redirected** — a hiring decision, ranking, opaque score or protected
      attribute.  The assistant explains what it *can* do instead.
    * **allowed** — normal evidence-seeking question.
    """
    text = (query or "").strip()
    if not text:
        return GuardrailDecision(allowed=False, blocked=True, reason="empty_query",
                                 message="Please enter a question.")

    if len(text) > max_chars:
        return GuardrailDecision(
            allowed=False,
            blocked=True,
            reason="input_too_long",
            message=f"That question is {len(text):,} characters; the limit is {max_chars:,}. Please shorten it.",
            categories=["input_too_long"],
        )

    decision = GuardrailDecision()

    # 1. Injection / exfiltration / tool escalation — hard stop.
    scan = scan_for_injection(text)
    decision.injection = scan
    hard = scan.families & HARD_BLOCK_FAMILIES
    if hard:
        decision.allowed = False
        decision.blocked = True
        decision.reason = "prompt_injection"
        decision.categories = sorted(hard)
        decision.matched_text = [f.matched_text for f in scan.findings if f.family in hard][:5]
        decision.message = (
            "I can't act on instructions like that. I only answer questions about the documents "
            "you have supplied, I never reveal my configuration or credentials, and I never run "
            "commands or take actions outside this app."
        )
        return decision

    # 2. Protected / sensitive attributes.
    protected_hits = [kind for pattern, kind in _COMPILED_PROTECTED if pattern.search(text)]
    if protected_hits:
        decision.allowed = False
        decision.redirected = True
        decision.reason = "protected_attribute"
        decision.categories = sorted(set(protected_hits))
        decision.message = PROTECTED_ATTRIBUTE_MESSAGE
        return decision

    # 3. Hiring decisions, rankings and opaque scores.
    decision_hits = [kind for pattern, kind in _COMPILED_DECISION if pattern.search(text)]
    soft_hits = [kind for pattern, kind in _COMPILED_SOFT if pattern.search(text)]
    if decision_hits or soft_hits:
        decision.allowed = False
        decision.redirected = True
        decision.reason = "out_of_scope_decision"
        decision.categories = sorted(set(decision_hits + soft_hits))
        decision.message = OUT_OF_SCOPE_MESSAGE
        return decision

    return decision


# --------------------------------------------------------------------------
# Untrusted-content neutralisation
# --------------------------------------------------------------------------

#: Wrapper used when untrusted text is placed in a prompt.  The model is told
#: that everything inside is data.  We also strip the closing marker from the
#: payload so document text cannot fake its own end-of-data boundary.
UNTRUSTED_OPEN = "<<<UNTRUSTED_DOCUMENT_DATA>>>"
UNTRUSTED_CLOSE = "<<<END_UNTRUSTED_DOCUMENT_DATA>>>"

_MARKER_RE = re.compile(
    r"<<<\s*(?:END_)?UNTRUSTED_DOCUMENT_DATA\s*>>>", re.IGNORECASE
)


def neutralise_untrusted_text(text: str, *, scan: InjectionScan | None = None) -> str:
    """Make document text safe to embed in a prompt.

    * Removes any text that mimics our own untrusted-data delimiters.
    * Rewrites imperative injection phrases into quoted, inert form so that a
      model skimming the context does not read them as commands.
    """
    if not text:
        return ""

    cleaned = _MARKER_RE.sub("[redacted-marker]", text)

    scan = scan or scan_for_injection(cleaned)
    if not scan.detected:
        return cleaned

    # Replace each flagged span with an inert, visible marker.  Offsets are
    # applied in reverse so earlier spans stay valid.
    for finding in sorted(scan.findings, key=lambda f: f.span[0], reverse=True):
        start, end = finding.span
        if 0 <= start < end <= len(cleaned):
            cleaned = (
                cleaned[:start]
                + f"[neutralised {finding.kind}: text quoted, not followed]"
                + cleaned[end:]
            )
    return cleaned


def scan_untrusted_text(text: str) -> InjectionScan:
    """Convenience re-export so callers only import from ``safety``."""
    return scan_for_injection(text)


def wrap_untrusted(text: str) -> str:
    """Wrap already-neutralised text in the untrusted-data envelope."""
    return f"{UNTRUSTED_OPEN}\n{neutralise_untrusted_text(text)}\n{UNTRUSTED_CLOSE}"


__all__ = [
    "OUT_OF_SCOPE_MESSAGE",
    "PROTECTED_ATTRIBUTE_MESSAGE",
    "UNTRUSTED_CLOSE",
    "UNTRUSTED_OPEN",
    "GuardrailDecision",
    "check_query",
    "neutralise_untrusted_text",
    "scan_untrusted_text",
    "wrap_untrusted",
]
