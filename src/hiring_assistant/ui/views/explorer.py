"""Candidate Explorer — cited summaries and the requirement evidence matrix.

Deliberately contains no aggregate score, no ranking and no recommendation.
The most prominent element on the page is the explanation of what a "not found"
status does and does not mean.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from ...errors import HiringAssistantError
from ...utils import truncate
from ..components import (
    banner,
    citation_list,
    download_row,
    empty_state,
    error_panel,
    esc,
    pill,
    requirement_row,
    section_label,
    tiles,
)
from ..theme import COLORS, plotly_layout, status_color

NOT_FOUND_EXPLANATION = (
    "<strong>What “Not found in supplied documents” means.</strong> It means no passage in the "
    "documents available here provides evidence for that requirement. It is <em>not</em> a "
    "statement that the candidate lacks the skill — the profile may simply not mention it, or the "
    "document may not have been supplied. Treat it as a prompt to ask the candidate or to request "
    "more documents."
)


def render(app: Any) -> None:
    st.markdown("## Candidate Explorer")
    st.caption(
        "Read an evidence summary of a candidate's own documents, and map a job description's "
        "requirements against those documents. This view shows evidence only — it produces no "
        "suitability score, no ranking and no hire/reject recommendation."
    )

    if not app.index_loaded:
        empty_state("No index yet", "Load the synthetic corpus from the Knowledge Base tab first.")
        return

    candidates = app.candidates()
    jobs = app.jobs()

    if not candidates:
        empty_state(
            "No candidate profiles found",
            "Load the synthetic corpus from the Knowledge Base tab, or upload candidate documents.",
        )
        return

    banner("info", NOT_FOUND_EXPLANATION)

    # -- selection --------------------------------------------------------
    left, right = st.columns([1, 1])
    with left:
        candidate_labels = {c.candidate_id: c.label() for c in candidates}
        candidate_id = st.selectbox(
            f"Candidate ({len(candidates)} available)",
            options=list(candidate_labels),
            format_func=lambda key: candidate_labels[key],
            key="explorer_candidate",
        )
    with right:
        if jobs:
            job_labels = {j.job_id: j.label() for j in jobs}
            job_id = st.selectbox(
                f"Job description ({len(jobs)} available)",
                options=list(job_labels),
                format_func=lambda key: job_labels[key],
                key="explorer_job",
            )
        else:
            job_id = None
            st.info("No job descriptions are indexed. Upload one to build a requirement matrix.")

    candidate = next(c for c in candidates if c.candidate_id == candidate_id)
    job = next((j for j in jobs if j.job_id == job_id), None) if job_id else None

    st.divider()

    # -- summary ----------------------------------------------------------
    _render_summary(app, candidate)

    # -- matrix -----------------------------------------------------------
    if job is not None:
        st.divider()
        _render_matrix(app, candidate, job)


# --------------------------------------------------------------------------


def _render_summary(app: Any, candidate: Any) -> None:
    st.markdown(f"### {esc(candidate.name)}")
    st.caption(f"Source document: `{esc(candidate.filename)}` · {candidate.chunk_count} indexed passage(s)")

    use_llm = bool(app.generator.llm) and st.toggle(
        "Draft the summary with the language model",
        value=False,
        key="explorer_use_llm",
        help=(
            "Off: sections are extracted verbatim from the candidate's own document, so every line "
            "is a direct quotation. On: the model drafts prose, still constrained to cited passages."
        ),
    )

    try:
        summary = app.summarise_candidate(candidate, use_llm=use_llm)
    except HiringAssistantError as error:
        error_panel(error)
        return
    except Exception as error:  # pragma: no cover
        error_panel(error)
        return

    tiles(
        [
            {"label": "Core skills listed", "value": f"{len(summary.skills)}"},
            {"label": "Additional skills", "value": f"{len(summary.additional_skills)}"},
            {"label": "Projects described", "value": f"{len(summary.project_lines)}"},
            {
                "label": "Summary mode",
                "value": "Model-drafted" if summary.mode == "llm" else "Verbatim extraction",
                "detail": "no unsupported claims either way",
            },
        ]
    )

    if summary.headline:
        st.markdown(f"**{esc(summary.headline)}**")

    if summary.narrative:
        st.markdown(summary.narrative)

    left, right = st.columns(2)
    with left:
        if summary.skills:
            section_label("Core skills (as stated)")
            st.markdown(" ".join(pill(skill, "info") for skill in summary.skills), unsafe_allow_html=True)
        if summary.additional_skills:
            section_label("Additional skills (as stated)")
            st.markdown(
                " ".join(pill(skill, "muted") for skill in summary.additional_skills),
                unsafe_allow_html=True,
            )
    with right:
        if summary.experience_lines:
            section_label("Experience (verbatim)")
            for line in summary.experience_lines:
                st.markdown(f"- {esc(line)}")
        if summary.project_lines:
            section_label("Selected projects (verbatim)")
            for line in summary.project_lines[:5]:
                st.markdown(f"- {esc(line)}")

    st.caption(
        "Everything above is taken from the candidate's own document. No experience, "
        "qualification, employer or date has been inferred, and no protected or personal "
        "characteristic has been assessed."
    )

    if summary.citations:
        with st.expander(f"Source passages ({len(summary.citations)})", expanded=False):
            citation_list(summary.citations)


# --------------------------------------------------------------------------


def _render_matrix(app: Any, candidate: Any, job: Any) -> None:
    st.markdown(f"### Requirement evidence — {esc(job.title)}")
    st.caption(
        f"Each explicit requirement in `{esc(job.filename)}` is matched against "
        f"{esc(candidate.name)}'s documents."
    )

    with st.expander("Requirements extracted from this job description", expanded=False):
        try:
            requirements = app.job_requirements(job)
        except Exception as error:  # pragma: no cover
            requirements = []
            error_panel(error)
        if requirements:
            for index, requirement in enumerate(requirements, start=1):
                st.markdown(f"{index}. {esc(requirement)}")
            st.caption(
                "Only requirements the document states explicitly are used. Nothing is inferred "
                "from the job title or from domain knowledge."
            )
        else:
            st.info(
                "No explicit requirements could be extracted from this job description. The matrix "
                "is left empty rather than guessed."
            )

    use_llm = bool(app.generator.llm) and st.toggle(
        "Refine statuses with the language model",
        value=False,
        key="matrix_use_llm",
        help=(
            "Off: statuses come from a deterministic, reproducible rule. On: the model proposes a "
            "status, which is still validated against retrieved citations."
        ),
    )

    with st.spinner("Mapping requirements to evidence…"):
        try:
            matrix = app.build_matrix(candidate, job, use_llm=use_llm)
        except HiringAssistantError as error:
            error_panel(error)
            return
        except Exception as error:  # pragma: no cover
            error_panel(error)
            return

    counts = matrix.counts()
    tiles(
        [
            {"label": "Requirements", "value": f"{len(matrix.rows)}"},
            {"label": "Supported", "value": f"{counts['supported']}", "tone": "good"},
            {"label": "Partial", "value": f"{counts['partial']}", "tone": "warn"},
            {"label": "Not found", "value": f"{counts['not_found']}"},
        ]
    )

    if not matrix.rows:
        return

    # -- visualisation ----------------------------------------------------
    left, right = st.columns([1, 1.15])
    with left:
        _status_donut(counts)
    with right:
        _coverage_chart(matrix)

    st.caption(
        "These charts show the *distribution of document evidence*, not candidate quality. "
        "There is deliberately no combined score, because a single number would hide which "
        "requirements are supported and which are simply absent from the documents."
    )

    # -- rows -------------------------------------------------------------
    section_label("Requirement by requirement")
    for index, row in enumerate(matrix.rows, start=1):
        requirement_row(row, index)

    for warning in matrix.warnings:
        banner("info", esc(warning))

    # -- export -----------------------------------------------------------
    section_label("Export")
    download_row(
        [
            {
                "label": "Evidence matrix (.md)",
                "data": app.export_matrix_markdown(matrix),
                "filename": f"evidence_{matrix.candidate_id}_{matrix.job_id}.md",
                "mime": "text/markdown",
            },
            {
                "label": "Evidence matrix (.csv)",
                "data": app.export_matrix_csv(matrix),
                "filename": f"evidence_{matrix.candidate_id}_{matrix.job_id}.csv",
                "mime": "text/csv",
            },
        ]
    )

    st.caption(
        "This matrix supports human evidence review. It does not replace it, and it must not be "
        "used as an automated accept/reject decision."
    )


def _status_donut(counts: dict[str, int]) -> None:
    """Evidence status split.  Uses the reserved status palette."""
    labels = ["Supported", "Partial", "Not found"]
    keys = ["supported", "partial", "not_found"]
    values = [counts.get(key, 0) for key in keys]

    if sum(values) == 0:
        return

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.info("Install plotly to see the evidence charts.")
        return

    figure = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            marker={"colors": [status_color(k) for k in keys], "line": {"color": "#FFFFFF", "width": 2}},
            textinfo="value",
            textfont={"size": 13, "color": "#FFFFFF"},
            hovertemplate="%{label}: %{value} requirement(s)<extra></extra>",
            sort=False,
            direction="clockwise",
        )
    )
    layout = plotly_layout(height=280, showlegend=True)
    layout["legend"] = {
        "orientation": "h",
        "yanchor": "bottom",
        "y": -0.12,
        "xanchor": "center",
        "x": 0.5,
        "font": {"size": 12},
    }
    layout["margin"] = {"l": 8, "r": 8, "t": 8, "b": 8}
    layout["title"] = {"text": "Evidence status distribution", "font": {"size": 13, "color": COLORS["navy_900"]}}
    figure.update_layout(**layout)
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def _coverage_chart(matrix: Any) -> None:
    """Per-requirement ranking score, coloured by status.

    A bar chart is the right form here: each requirement is a distinct category
    being compared on one measure. Colour carries the status so a reader can see
    at a glance which requirements are unsupported.
    """
    rows = matrix.rows
    if not rows:
        return

    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    labels = [truncate(row.requirement, 46) for row in rows]
    scores = [round(row.best_score, 3) for row in rows]
    colors = [status_color(row.status) for row in rows]
    texts = [row.status_label for row in rows]

    figure = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker={"color": colors},
            text=[f"{score:.2f}" for score in scores],
            textposition="outside",
            textfont={"size": 11, "color": COLORS["slate_500"]},
            customdata=texts,
            hovertemplate="%{y}<br>Ranking score: %{x:.3f}<br>Status: %{customdata}<extra></extra>",
        )
    )
    layout = plotly_layout(height=max(260, 30 * len(rows) + 70))
    layout["margin"] = {"l": 8, "r": 40, "t": 32, "b": 8}
    layout["xaxis"] = {
        "title": {"text": "Best matching passage — ranking score", "font": {"size": 11, "color": COLORS["slate_500"]}},
        "range": [0, 1.05],
        "showgrid": True,
        "gridcolor": COLORS["slate_100"],
        "tickfont": {"size": 11, "color": COLORS["slate_500"]},
    }
    layout["yaxis"] = {
        "automargin": True,
        "tickfont": {"size": 11, "color": COLORS["slate_700"]},
        "autorange": "reversed",
    }
    layout["title"] = {
        "text": "Evidence strength per requirement",
        "font": {"size": 13, "color": COLORS["navy_900"]},
    }
    figure.update_layout(**layout)
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


__all__ = ["render"]
