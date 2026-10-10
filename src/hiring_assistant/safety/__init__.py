"""Responsible-use guardrails.

Three distinct concerns live here and they are deliberately kept separate:

1. **Untrusted-content handling** — text inside a document is data.  If it
   contains something that looks like an instruction, we never execute it; we
   flag it, neutralise it in the prompt and surface it to the user.
2. **Out-of-scope requests** — the product retrieves and summarises evidence.
   It does not rank candidates, score suitability or make hiring decisions.
3. **Protected attributes** — we refuse to infer or evaluate sensitive traits
   even when a document happens to mention them.
"""

from .guardrails import (
    OUT_OF_SCOPE_MESSAGE,
    GuardrailDecision,
    check_query,
    neutralise_untrusted_text,
    scan_untrusted_text,
)
from .injection import InjectionFinding, scan_for_injection

__all__ = [
    "OUT_OF_SCOPE_MESSAGE",
    "GuardrailDecision",
    "InjectionFinding",
    "check_query",
    "neutralise_untrusted_text",
    "scan_for_injection",
    "scan_untrusted_text",
]
