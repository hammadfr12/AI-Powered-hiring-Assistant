"""Reusable UI components.

Rendering lives here rather than in the views so that a citation card, an
evidence row or an error panel looks and behaves identically wherever it
appears.  All components emit HTML built from escaped values — document text is
untrusted, so nothing from the corpus is ever interpolated raw.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Sequence
from typing import Any

import streamlit as st

from ..errors import HiringAssistantError
from ..models import Citation, RequirementEvidence, RetrievedChunk
from ..utils import truncate
from .theme import COLORS, PLOTLY_CONFIG

# --------------------------------------------------------------------------
# Escaping
# --------------------------------------------------------------------------


def esc(value: Any) -> str:
    """HTML-escape any value before it reaches the browser.

    Retrieved passages and filenames are untrusted input; a resume containing
    markup must render as text, not execute.
    """
    return html.escape(str(value if value is not None else ""), quote=True)


#: Matches the citation markers the generator emits, e.g. ``[S3]``.
_MARKER_RE = re.compile(r"\[S(\d{1,3})\]")


def render_answer_html(text: str, *, tone: str = "normal") -> str:
    """Render answer text as HTML with citation markers styled as chips.

    Handles a small, deliberate subset of Markdown — paragraphs, bullet lists,
    bold and inline code — because the extractive generator emits exactly that
    and pulling in a full Markdown engine for it would be overkill.
    """
    css_class = {"normal": "answer", "abstained": "answer abstained", "blocked": "answer blocked"}.get(
        tone, "answer"
    )

    blocks: list[str] = []
    for raw_block in re.split(r"\n\s*\n", text or ""):
        block = raw_block.strip()
        if not block:
            continue

        lines = block.split("\n")
        if all(line.strip().startswith(("- ", "* ", "• ")) for line in lines if line.strip()):
            items = "".join(
                f"<li>{_inline(line.strip().lstrip('-*•').strip())}</li>" for line in lines if line.strip()
            )
            blocks.append(f"<ul>{items}</ul>")
            continue

        # A block that is entirely bullet lines with a trailing non-bullet line
        # (the extractive generator's closing note) is split.
        bullets = [line for line in lines if line.strip().startswith(("- ", "* ", "• "))]
        if bullets and len(bullets) >= len([line for line in lines if line.strip()]) - 1:
            items = "".join(
                f"<li>{_inline(line.strip().lstrip('-*•').strip())}</li>"
                for line in lines
                if line.strip() and line.strip().startswith(("- ", "* ", "• "))
            )
            tail = [
                line for line in lines if line.strip() and not line.strip().startswith(("- ", "* ", "• "))
            ]
            blocks.append(f"<ul>{items}</ul>")
            for line in tail:
                blocks.append(f"<p>{_inline(line)}</p>")
            continue

        blocks.append(f"<p>{_inline(' '.join(line.strip() for line in lines if line.strip()))}</p>")

    return f'<div class="{css_class}">' + "".join(blocks) + "</div>"


def _inline(text: str) -> str:
    """Escape, then apply bold / code / citation-chip formatting."""
    out = esc(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"`(.+?)`", r"<code>\1</code>", out)
    out = _MARKER_RE.sub(
        lambda m: f'<span class="cite" title="Source {m.group(1)} — open the citation below">'
        f"S{m.group(1)}</span>",
        out,
    )
    return out


# --------------------------------------------------------------------------
# Chrome
# --------------------------------------------------------------------------


def app_header(title: str, subtitle: str, mode_label: str, external: bool) -> None:
    badge = "External provider" if external else "Runs locally"
    st.markdown(
        f"""
<div class="app-header">
  <div class="mark">AI</div>
  <div class="titles">
    <div class="t">{esc(title)}</div>
    <div class="s">{esc(subtitle)}</div>
  </div>
  <div class="spacer"></div>
  <div class="mode">{esc(mode_label)}<br>{esc(badge)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def banner(kind: str, message: str, *, icon: str | None = None) -> None:
    """Coloured notice.  ``kind`` is one of info, warn, danger, ok."""
    icons = {"info": "ℹ", "warn": "⚠", "danger": "⛔", "ok": "✓"}
    st.markdown(
        f'<div class="banner {esc(kind)}"><span class="ico">{esc(icon or icons.get(kind, "•"))}</span>'
        f"<span>{message}</span></div>",
        unsafe_allow_html=True,
    )


def synthetic_notice() -> None:
    banner(
        "info",
        "<strong>Synthetic demo data.</strong> Every candidate profile, job description and "
        "policy document in this corpus is fictional and machine-generated. It contains no real "
        "personal data and no protected characteristics. "
        "This assistant retrieves and summarises evidence — it does not score, rank or select candidates.",
    )


def error_panel(error: HiringAssistantError | Exception) -> None:
    """Render a structured error with its remedy."""
    if isinstance(error, HiringAssistantError):
        label, message, remedy = error.label, error.message, error.remedy
    else:
        label, message, remedy = type(error).__name__, str(error), ""
    st.markdown(
        f'<div class="err"><div class="t">{esc(label)}</div>'
        f'<div class="m">{esc(message)}</div>'
        + (f'<div class="r">What to do: {esc(remedy)}</div>' if remedy else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def empty_state(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="empty"><div class="t">{esc(title)}</div>'
        f'<div class="s">{esc(subtitle)}</div></div>',
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{esc(text)}</div>', unsafe_allow_html=True)


def tiles(items: Sequence[dict[str, str]]) -> None:
    """Row of metric tiles.  Each item: {label, value, detail?, tone?}."""
    cells = []
    for item in items:
        tone = item.get("tone", "")
        cells.append(
            f'<div class="tile {esc(tone)}">'
            f'<div class="k">{esc(item.get("label", ""))}</div>'
            f'<div class="v">{esc(item.get("value", ""))}</div>'
            + (f'<div class="d">{esc(item["detail"])}</div>' if item.get("detail") else "")
            + "</div>"
        )
    st.markdown(f'<div class="tiles">{"".join(cells)}</div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "muted") -> str:
    """Return a pill's HTML (so it can be embedded in a larger block)."""
    return f'<span class="pill {esc(kind)}">{esc(text)}</span>'


def origin_pill(synthetic: bool) -> str:
    return pill("Synthetic demo data" if synthetic else "Uploaded document", "synthetic" if synthetic else "upload")


def score_bar(score: float, *, maximum: float = 1.0) -> str:
    """A relevance bar.  Labelled a *ranking signal*, never a probability."""
    pct = max(0.0, min(1.0, score / maximum if maximum else 0.0)) * 100
    return (
        f'<div class="score-wrap"><div class="score-bar">'
        f'<div class="score-fill" style="width:{pct:.1f}%"></div></div>'
        f'<span class="score-num">{score:.3f}</span></div>'
    )


# --------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------


def citation_card(citation: Citation, index: int, *, expanded: bool = False) -> None:
    """One citation, showing the exact supporting excerpt and its provenance."""
    st.markdown(
        f"""
<div class="cit-card">
  <div class="cit-head">
    <span class="cit-num">S{index}</span>
    <span class="cit-file">{esc(citation.filename)}</span>
    <span class="cit-loc">{esc(citation.locator)}</span>
    {origin_pill(citation.synthetic)}
  </div>
  <div class="cit-quote">{esc(truncate(citation.excerpt, 420))}</div>
  <div class="cit-meta">
    Collection: {esc(citation.collection)} · Relevance signal: {citation.score:.3f}
    (a ranking score, not a probability that the statement is true)
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def citation_list(citations: Sequence[Citation], *, heading: str = "Sources") -> None:
    if not citations:
        return
    section_label(f"{heading} ({len(citations)})")
    for index, citation in enumerate(citations, start=1):
        citation_card(citation, index)


# --------------------------------------------------------------------------
# Search results
# --------------------------------------------------------------------------


def source_row(item: RetrievedChunk, index: int) -> None:
    chunk = item.chunk
    terms = ", ".join(item.matched_terms[:8]) if item.matched_terms else "—"
    st.markdown(
        f"""
<div class="src">
  <div class="src-head">
    <div>
      <span class="cit-num">{index}</span>
      <span class="src-file">{esc(chunk.filename)}</span>
      <span class="cit-loc">{esc(chunk.locator)}</span>
    </div>
    {score_bar(item.score)}
  </div>
  <div class="src-snip">{esc(truncate(chunk.text, 340))}</div>
  <div class="src-terms">Collection: {esc(chunk.collection)} · Matched terms: {esc(terms)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def search_results(results: Iterable[RetrievedChunk]) -> None:
    for index, item in enumerate(results, start=1):
        source_row(item, index)


# --------------------------------------------------------------------------
# Requirement matrix
# --------------------------------------------------------------------------


def requirement_row(row: RequirementEvidence, index: int, *, show_evidence: bool = True) -> None:
    evidence_html = ""
    if show_evidence and row.citations:
        items = "".join(
            f'<div style="margin-bottom:.34rem">'
            f'<strong>{esc(c.filename)}</strong> '
            f'<span class="cit-loc">{esc(c.locator)}</span> · relevance {c.score:.3f}'
            f'<div style="color:{COLORS["slate_700"]};margin-top:.16rem">{esc(truncate(c.excerpt, 260))}</div>'
            f"</div>"
            for c in row.citations[:3]
        )
        evidence_html = f'<div class="req-e">{items}</div>'

    st.markdown(
        f"""
<div class="req {esc(row.status)}">
  <div class="req-q">{index}. {esc(row.requirement)}
    <span style="float:right">{pill(row.status_label, row.status)}</span>
  </div>
  <div class="req-r">{esc(row.rationale)}</div>
  {evidence_html}
</div>
""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def try_plotly(figure: Any, *, height: int | None = None) -> bool:
    """Render a Plotly figure, degrading gracefully if Plotly is unavailable."""
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError:
        return False
    if height is not None:
        figure.update_layout(height=height)
    st.plotly_chart(figure, use_container_width=True, config=PLOTLY_CONFIG)
    return True


def download_row(items: Sequence[dict[str, Any]]) -> None:
    """A row of download buttons.  Each item: {label, data, filename, mime}."""
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        with column:
            st.download_button(
                label=item["label"],
                data=item["data"],
                file_name=item["filename"],
                mime=item.get("mime", "text/plain"),
                use_container_width=True,
            )


__all__ = [
    "app_header",
    "banner",
    "citation_card",
    "citation_list",
    "download_row",
    "empty_state",
    "error_panel",
    "esc",
    "origin_pill",
    "pill",
    "render_answer_html",
    "requirement_row",
    "score_bar",
    "search_results",
    "section_label",
    "source_row",
    "synthetic_notice",
    "tiles",
    "try_plotly",
]
