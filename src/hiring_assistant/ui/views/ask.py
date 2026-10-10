"""Ask Assistant — cited Q&A with visible abstention."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ...errors import HiringAssistantError
from ...retrieval.search import SearchFilters
from ...safety.guardrails import check_query
from ..components import (
    banner,
    citation_list,
    download_row,
    empty_state,
    error_panel,
    esc,
    render_answer_html,
    section_label,
    source_row,
)

EXAMPLE_QUESTIONS = [
    "How many stages are in the standard interview process?",
    "What is the annual leave entitlement?",
    "How long are recruitment records retained after a process closes?",
    "What does the evidence policy say about what must not be recorded in notes?",
    "How many working days do analytics roles take from screen to offer?",
    "Which candidate profiles mention Apache Airflow?",
    "What is the company policy on sabbatical leave?",
]


def _filters_from_sidebar(available: dict[str, Any]) -> SearchFilters:
    """Read the shared filter widgets from session state."""
    return SearchFilters(
        collections=list(st.session_state.get("filter_collections", [])),
        file_types=list(st.session_state.get("filter_file_types", [])),
    )


def render(app: Any) -> None:
    st.markdown("## Ask Assistant")
    st.caption(
        "Ask a question in plain language. Answers are assembled only from the documents you "
        "have supplied, every claim carries a citation, and the assistant says so plainly when "
        "the corpus does not contain the answer."
    )

    if not app.index_loaded:
        empty_state(
            "No index yet",
            "Open the Knowledge Base tab and load the synthetic corpus to get started.",
        )
        return

    available = app.search_service.available_filters()
    filters = _filters_from_sidebar(available)

    # -- session ----------------------------------------------------------
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"sess_{st.session_state.get('_run', 0)}"
    session_id = st.session_state.session_id

    history = app.history(session_id)
    turn_count = len([m for m in history if m.role == "user"])

    controls = st.columns([1, 1, 1, 1.4])
    with controls[0]:
        st.metric("Turns in this session", turn_count)
    with controls[1]:
        st.metric("Scope", "Filtered" if filters.active else "All documents")
    with controls[2]:
        st.metric("Answer mode", "Language model" if app.generator.llm else "Extractive")
    with controls[3]:
        st.write("")
        if st.button("Clear conversation", use_container_width=True, disabled=turn_count == 0):
            removed = app.clear_history(session_id)
            st.session_state.pop("last_result", None)
            st.toast(f"Cleared {removed} message(s). Conversational context is now empty.")
            st.rerun()

    if turn_count:
        st.caption(
            "Conversation memory is scoped to this session and is used only to resolve follow-up "
            "wording. It is never treated as evidence for a factual claim."
        )

    # -- transcript -------------------------------------------------------
    if history:
        with st.expander(f"Conversation transcript ({len(history)} messages)", expanded=False):
            for message in history:
                role = "You" if message.role == "user" else "Assistant"
                st.markdown(f"**{esc(role)}** — {esc(message.content[:600])}")
                if message.cited_chunk_ids:
                    st.caption(f"Cited {len(message.cited_chunk_ids)} passage(s).")

    st.divider()

    # -- examples ---------------------------------------------------------
    if turn_count == 0:
        section_label("Try one of these")
        columns = st.columns(2)
        for index, question in enumerate(EXAMPLE_QUESTIONS[:6]):
            with columns[index % 2]:
                if st.button(question, key=f"ex_{index}", use_container_width=True):
                    st.session_state.pending_question = question
                    st.rerun()

    # -- input ------------------------------------------------------------
    question = st.chat_input("Ask about candidates, roles or hiring documents…")
    if st.session_state.get("pending_question"):
        question = st.session_state.pop("pending_question")

    if question:
        st.session_state.last_question = question
        _answer(app, question, filters, session_id)

    elif st.session_state.get("last_result") and not question:
        _render_result(st.session_state.last_result)


def _answer(app: Any, question: str, filters: SearchFilters, session_id: str) -> None:
    st.markdown(f"**You** — {esc(question)}")

    # Screen before spending any retrieval work, so a refused request is
    # visibly instant and obviously not a retrieval failure.
    decision = check_query(question)
    if decision.blocked or decision.redirected:
        with st.spinner("Checking the request…"):
            result = app.ask(question, filters=filters, session_id=session_id)
        st.session_state.last_result = result
        _render_result(result)
        return

    with st.spinner("Retrieving passages and drafting a grounded answer…"):
        try:
            result = app.ask(question, filters=filters, session_id=session_id)
        except HiringAssistantError as error:
            error_panel(error)
            return
        except Exception as error:  # pragma: no cover - defensive
            error_panel(error)
            return

    st.session_state.last_result = result
    _render_result(result)


def _render_result(result: Any) -> None:
    answer = result.answer

    # -- mode and provenance ---------------------------------------------
    meta = st.columns(4)
    with meta[0]:
        st.metric("Latency", f"{answer.latency_ms:,.0f} ms")
    with meta[1]:
        st.metric("Citations", len(answer.citations))
    with meta[2]:
        st.metric("Passages searched", len(answer.retrieved))
    with meta[3]:
        st.metric("Answer mode", "Extractive" if answer.mode == "extractive" else "Language model")

    # -- guardrail outcomes ----------------------------------------------
    if answer.blocked:
        if answer.block_reason == "prompt_injection":
            banner("danger", esc(answer.answer_text))
            st.caption(
                "This request tried to override the assistant's instructions or extract "
                "configuration. The assistant answers questions about your documents; it never "
                "follows instructions found in text, and never reveals prompts or credentials."
            )
        elif answer.block_reason == "out_of_scope_decision":
            banner("warn", esc(answer.answer_text))
            st.caption(
                "Evidence review is a human task. The Candidate Explorer tab maps each job "
                "requirement to the documents that do or do not support it, without producing a "
                "score or a recommendation."
            )
        elif answer.block_reason == "protected_attribute":
            banner("warn", esc(answer.answer_text))
        else:
            banner("warn", esc(answer.answer_text))
        st.markdown("---")
        return

    # -- abstention -------------------------------------------------------
    if answer.abstained:
        banner("warn", "The supplied documents do not contain enough evidence to answer this.")
        st.markdown(render_answer_html(answer.answer_text, tone="abstained"), unsafe_allow_html=True)
        st.caption(
            "Abstaining is the correct behaviour here, not a failure. The assistant will not "
            "answer from general knowledge, because that would risk stating something the "
            "documents do not support."
        )
        if answer.retrieved:
            with st.expander(
                f"Closest passages that were considered ({len(answer.retrieved)})", expanded=False
            ):
                st.caption(
                    "These are the nearest matches. They are shown so you can judge for yourself "
                    "whether the corpus covers the question under different wording."
                )
                for index, item in enumerate(answer.retrieved, start=1):
                    source_row(item, index)
        st.markdown("---")
        return

    # -- answer -----------------------------------------------------------
    st.markdown(render_answer_html(answer.answer_text), unsafe_allow_html=True)

    for warning in answer.warnings:
        banner("warn", esc(warning))

    if answer.invalid_citations:
        banner(
            "warn",
            f"{len(answer.invalid_citations)} reference(s) in the draft pointed at passages that "
            "were not retrieved and were removed before display.",
        )

    if answer.citations:
        citation_list(answer.citations, heading="Sources — open to see the exact excerpt")
        st.caption(
            "Every citation above resolved to a passage that was actually retrieved for this "
            "question. The relevance figure is a ranking signal, not a probability that the "
            "statement is true."
        )
    else:
        banner("info", "This answer was produced without citations, so treat it with caution.")

    # -- exports ----------------------------------------------------------
    section_label("Export")
    download_row(
        [
            {
                "label": "Cited answer (.md)",
                "data": app.export_answer_markdown(result),
                "filename": "cited_answer.md",
                "mime": "text/markdown",
            },
            {
                "label": "Retrieved passages (.md)",
                "data": _passages_markdown(answer),
                "filename": "retrieved_passages.md",
                "mime": "text/markdown",
            },
        ]
    )


def _passages_markdown(answer: Any) -> str:
    lines = [
        "# Retrieved passages",
        "",
        f"**Question:** {answer.question}",
        f"**Passages:** {len(answer.retrieved)}",
        "",
    ]
    for index, item in enumerate(answer.retrieved, start=1):
        chunk = item.chunk
        origin = "synthetic demo data" if chunk.synthetic else "uploaded document"
        lines += [
            f"## {index}. {chunk.filename} — {chunk.locator}",
            "",
            f"- Collection: `{chunk.collection}`",
            f"- Origin: {origin}",
            f"- Relevance signal: {item.score:.4f}",
            f"- Matched terms: {', '.join(item.matched_terms) or '—'}",
            "",
            f"> {chunk.text.strip()}",
            "",
        ]
    return "\n".join(lines)


__all__ = ["render"]
