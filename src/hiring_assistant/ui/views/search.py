"""Semantic Search — ranked passages with provenance, no generation."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ...errors import HiringAssistantError
from ...retrieval.search import SearchFilters
from ..components import (
    banner,
    download_row,
    empty_state,
    error_panel,
    esc,
    section_label,
    source_row,
    tiles,
)

EXAMPLES = [
    "data visualisation and dashboards",
    "Apache Airflow orchestration",
    "incident response and on-call",
    "financial modelling and forecasting",
    "accessibility WCAG",
    "settlement and reconciliation",
]


def render(app: Any) -> None:
    st.markdown("## Semantic Search")
    st.caption(
        "Retrieve passages by meaning rather than exact keywords. Results are ranked by a blend "
        "of embedding similarity and term overlap. The score is a ranking signal — it is not a "
        "probability that a passage is relevant, and it is not a measure of truth."
    )

    if not app.index_loaded:
        empty_state("No index yet", "Load the synthetic corpus from the Knowledge Base tab first.")
        return

    available = app.search_service.available_filters()

    query = st.text_input(
        "Search the corpus",
        placeholder="e.g. dashboard reporting for business stakeholders",
        key="search_query",
    )

    row = st.columns([1, 1, 1])
    with row[0]:
        top_k = st.slider("Passages to return", 1, 20, app.settings.top_k, key="search_topk")
    with row[1]:
        min_relevance = st.slider(
            "Minimum relevance",
            0.0,
            0.9,
            float(app.settings.min_relevance),
            step=0.01,
            key="search_minrel",
            help="Passages below this ranking score are dropped before display.",
        )
    with row[2]:
        st.write("")
        st.write("")
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    if not query:
        section_label("Example searches")
        columns = st.columns(3)
        for index, example in enumerate(EXAMPLES):
            with columns[index % 3]:
                if st.button(example, key=f"sex_{index}", use_container_width=True):
                    st.session_state.search_query = example
                    st.rerun()
        return

    if not search_clicked and st.session_state.get("search_last") != query:
        # Allow the example buttons and the first render to trigger a search.
        search_clicked = True
    st.session_state.search_last = query

    filters = SearchFilters(
        collections=list(st.session_state.get("filter_collections", [])),
        file_types=list(st.session_state.get("filter_file_types", [])),
    )

    try:
        result = app.search(query, top_k=top_k, filters=filters, min_relevance=min_relevance)
    except HiringAssistantError as error:
        error_panel(error)
        return
    except Exception as error:  # pragma: no cover - defensive
        error_panel(error)
        return

    tiles(
        [
            {"label": "Passages returned", "value": f"{len(result.results)}", "tone": "accent"},
            {
                "label": "Best ranking score",
                "value": f"{result.best_score:.3f}",
                "detail": "ranking signal, not probability",
            },
            {"label": "Candidates examined", "value": f"{result.pool_size}"},
            {"label": "Time", "value": f"{result.duration_ms:,.0f} ms"},
        ]
    )

    if not result.results:
        banner("warn", esc(result.message or "No passages matched this query."))
        return

    if result.low_evidence:
        banner("warn", esc(result.message))
        if result.absent_terms:
            st.caption(
                "Words in your query that do not appear anywhere in the corpus: "
                + ", ".join(result.absent_terms)
            )

    if filters.active:
        banner("info", f"Filters applied — {esc(filters.describe())}")

    section_label(f"Results ({len(result.results)})")
    for index, item in enumerate(result.results, start=1):
        source_row(item, index)

    with st.expander("How ranking works", expanded=False):
        st.markdown(
            """
**Score = 0.72 × embedding similarity + 0.28 × term overlap.**

- *Embedding similarity* is cosine similarity between your query's vector and the passage's,
  produced by the configured embedding model. It captures paraphrase — searching
  "data visualisation" finds passages that say "Data Visualization".
- *Term overlap* is the share of your query's content words that appear in the passage. It
  catches exact terms the embedding model may under-weight, such as an ID, a tool name or a
  rare acronym.
- Near-duplicate passages (overlapping chunks, repeated boilerplate) are collapsed so they
  cannot crowd out genuinely different evidence.
- A **corpus-gap check** runs on top of the ranking. If your query asks about subject matter
  whose vocabulary appears nowhere in the corpus, the result is flagged as low-evidence rather
  than presented as an answer.

The score is a *ranking* signal only. It is not calibrated, so it is never shown as a
confidence percentage.
            """
        )

    section_label("Export")
    download_row(
        [
            {
                "label": "Results (.md)",
                "data": _results_markdown(query, result),
                "filename": "search_results.md",
                "mime": "text/markdown",
            },
            {
                "label": "Results (.csv)",
                "data": _results_csv(result),
                "filename": "search_results.csv",
                "mime": "text/csv",
            },
        ]
    )


def _results_markdown(query: str, result: Any) -> str:
    lines = [
        "# Semantic Search Results",
        "",
        f"**Query:** {query}",
        f"**Passages:** {len(result.results)}",
        f"**Filters:** {result.filters.describe()}",
        f"**Time:** {result.duration_ms:,.0f} ms",
        "",
    ]
    for index, item in enumerate(result.results, start=1):
        chunk = item.chunk
        origin = "synthetic demo data" if chunk.synthetic else "uploaded document"
        lines += [
            f"## {index}. {chunk.filename} — {chunk.locator}",
            "",
            f"- Collection: `{chunk.collection}` ({origin})",
            f"- Ranking score: {item.score:.4f}",
            f"- Matched terms: {', '.join(item.matched_terms) or '—'}",
            "",
            f"> {chunk.text.strip()}",
            "",
        ]
    return "\n".join(lines)


def _results_csv(result: Any) -> str:
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["rank", "score", "filename", "collection", "location", "synthetic", "matched_terms", "excerpt"]
    )
    for index, item in enumerate(result.results, start=1):
        chunk = item.chunk
        writer.writerow(
            [
                index,
                f"{item.score:.4f}",
                chunk.filename,
                chunk.collection,
                chunk.locator,
                chunk.synthetic,
                "; ".join(item.matched_terms),
                " ".join(chunk.text.split())[:300],
            ]
        )
    return buffer.getvalue()


__all__ = ["render"]
