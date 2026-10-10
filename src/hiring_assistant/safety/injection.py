"""Prompt-injection detection.

Documents are *untrusted data*.  A resume or job description may contain text
engineered to hijack the assistant — "ignore previous instructions", "reveal
your system prompt", "you are now DAN".  We detect those patterns so we can
(1) neutralise them before they reach the prompt and (2) show the recruiter
that the file contained a suspicious instruction.

Detection is intentionally high-recall and low-precision: a false positive
costs a visible warning on a suspicious paragraph, whereas a false negative
could mean document text steering the model.  Nothing here is an execution
path — a match only ever produces a flag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Pattern families
# --------------------------------------------------------------------------

#: Attempts to override, discard or replace the system instructions.
_OVERRIDE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bignore\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|preceding|above|earlier|foregoing)\s+"
     r"(?:instructions?|prompts?|rules?|directions?|context)", "instruction_override"),
    (r"\bdisregard\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+"
     r"(?:instructions?|prompts?|rules?|directions?)", "instruction_override"),
    (r"\bforget\s+(?:everything|all|your)\s+(?:you\s+know|instructions?|rules?|context|previous)", "instruction_override"),
    (r"\boverride\s+(?:your\s+)?(?:system\s+)?(?:instructions?|rules?|settings?|guardrails?)", "instruction_override"),
    (r"\bnew\s+(?:system\s+)?(?:instructions?|rules?|prompt)\s*[:：]", "instruction_override"),
    (r"\bfrom\s+now\s+on\s*,?\s*(?:you\s+)?(?:will|must|should)\b", "instruction_override"),
)

#: Attempts to exfiltrate the system prompt, secrets or configuration.
_EXFIL_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:reveal|show|print|display|repeat|output|tell\s+me|give\s+me)\b[^.\n]{0,40}"
     r"\b(?:system\s+prompt|hidden\s+prompt|initial\s+prompt|your\s+instructions?|"
     r"prompt\s+template|system\s+message)", "prompt_exfiltration"),
    # Direct questions about the prompt, however phrased: "What is your system
    # prompt?", "What were your original instructions?".
    (r"\bwhat\s+(?:is|are|was|were)\s+(?:your|the)\s+(?:\w+\s+){0,3}"
     r"(?:system\s+prompt|instructions?|prompt|rules?|guidelines?)\b", "prompt_exfiltration"),
    (r"\b(?:your|the)\s+system\s+prompt\b", "prompt_exfiltration"),
    (r"\b(?:reveal|show|print|leak|expose|dump)\b[^.\n]{0,30}"
     r"\b(?:api[\s_-]?key|secret|token|password|credential|env(?:ironment)?\s+var)", "secret_exfiltration"),
    (r"\b(?:print|cat|read|output)\b\s+(?:the\s+)?(?:contents?\s+of\s+)?(?:\.env|/etc/passwd|id_rsa)", "secret_exfiltration"),
)

#: Attempts to escalate into actions the assistant must never take.
_ACTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:run|execute|eval)\s+(?:this\s+|the\s+following\s+)?"
     r"(?:code|script|command|shell|bash|python)\b", "tool_escalation"),
    (r"\b(?:curl|wget|rm\s+-rf|sudo|chmod\s+\+x|subprocess|os\.system)\b", "tool_escalation"),
    (r"\b(?:send|email|post|upload|exfiltrate)\b[^.\n]{0,40}"
     r"\b(?:to|at)\s+(?:https?://|[\w.-]+@[\w.-]+)", "data_exfiltration"),
    (r"\bdelete\s+(?:all\s+)?(?:the\s+)?(?:files?|documents?|index|database)\b", "destructive_action"),
)

#: Attempts to swap the assistant's identity or jailbreak it.
_ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\byou\s+are\s+now\s+(?:a|an|the)\b", "role_hijack"),
    (r"\bpretend\s+(?:to\s+be|you\s+are)\b", "role_hijack"),
    (r"\bact\s+as\s+(?:a|an|the|if)\b", "role_hijack"),
    (r"\b(?:jailbreak|DAN\s+mode|developer\s+mode|god\s+mode|sudo\s+mode|unrestricted\s+mode)\b", "jailbreak"),
    (r"\b(?:with\s+|any\s+)?no\s+(?:restrictions?|rules?|limits?|filters?|guidelines?)\b", "jailbreak"),
    (r"\bbypass\s+(?:your\s+)?(?:safety|guardrails?|filters?|restrictions?)\b", "jailbreak"),
)

#: Delimiter injection — fake chat turns or fake system tags inside a document.
_DELIMITER_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"<\|?(?:im_start|im_end|system|assistant|endoftext)\|?>", "delimiter_injection"),
    (r"^\s*(?:system|assistant)\s*:\s*", "delimiter_injection"),
    (r"\[/?(?:INST|SYS|SYSTEM)\]", "delimiter_injection"),
    (r"#{2,}\s*(?:system|instructions?)\b", "delimiter_injection"),
)

_ALL_FAMILIES: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("instruction_override", _OVERRIDE_PATTERNS),
    ("prompt_exfiltration", _EXFIL_PATTERNS),
    ("tool_escalation", _ACTION_PATTERNS),
    ("role_hijack", _ROLE_PATTERNS),
    ("delimiter_injection", _DELIMITER_PATTERNS),
)

#: Families that, if matched in a *user query*, mean we decline outright.
HARD_BLOCK_FAMILIES: frozenset[str] = frozenset(
    {"prompt_exfiltration", "tool_escalation", "jailbreak", "role_hijack", "instruction_override"}
)

_COMPILED: tuple[tuple[str, tuple[tuple[re.Pattern[str], str], ...]], ...] = tuple(
    (
        family,
        tuple((re.compile(pattern, re.IGNORECASE | re.MULTILINE), kind) for pattern, kind in patterns),
    )
    for family, patterns in _ALL_FAMILIES
)


@dataclass
class InjectionFinding:
    """One suspicious span found in untrusted text."""

    kind: str
    family: str
    matched_text: str
    span: tuple[int, int]
    severity: str = "medium"

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "family": self.family,
            "matched_text": self.matched_text,
            "span": list(self.span),
            "severity": self.severity,
        }


@dataclass
class InjectionScan:
    """The result of scanning one piece of text."""

    findings: list[InjectionFinding] = field(default_factory=list)
    scanned_chars: int = 0

    @property
    def detected(self) -> bool:
        return bool(self.findings)

    @property
    def families(self) -> set[str]:
        return {f.family for f in self.findings}

    @property
    def summary(self) -> str:
        if not self.findings:
            return "No prompt-injection patterns detected."
        kinds = sorted({f.kind for f in self.findings})
        return (
            f"Detected {len(self.findings)} prompt-injection pattern(s) "
            f"of type: {', '.join(kinds)}. The text was treated as data and not followed."
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "detected": self.detected,
            "families": sorted(self.families),
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
        }


def scan_for_injection(text: str, *, max_findings: int = 25) -> InjectionScan:
    """Scan ``text`` for prompt-injection patterns.

    Never raises: any text, including malformed or enormous input, returns a
    scan object.  Scanning is capped so a pathological document cannot stall
    ingestion.
    """
    if not text:
        return InjectionScan()

    window = text[:200_000]
    findings: list[InjectionFinding] = []

    for family, patterns in _COMPILED:
        for pattern, kind in patterns:
            for match in pattern.finditer(window):
                severity = "high" if family in HARD_BLOCK_FAMILIES else "medium"
                findings.append(
                    InjectionFinding(
                        kind=kind,
                        family=family,
                        matched_text=match.group(0)[:200],
                        span=(match.start(), match.end()),
                        severity=severity,
                    )
                )
                if len(findings) >= max_findings:
                    return InjectionScan(findings=findings, scanned_chars=len(window))

    findings.sort(key=lambda f: f.span[0])
    return InjectionScan(findings=findings, scanned_chars=len(window))


__all__ = [
    "HARD_BLOCK_FAMILIES",
    "InjectionFinding",
    "InjectionScan",
    "scan_for_injection",
]
