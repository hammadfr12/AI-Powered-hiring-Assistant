"""Knowledge Base — upload, inventory, indexing status and corpus health."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ...config import COLLECTIONS
from ...errors import HiringAssistantError
from ...ingestion.parsers import SUPPORTED_EXTENSIONS
from ..components import (
    banner,
    download_row,
    empty_state,
    error_panel,
    esc,
    pill,
    section_label,
    tiles,
)
from ..theme import COLORS, SEQUENTIAL_BLUE, plotly_layout

STATUS_TONE = {
    "indexed": "supported",
    "pending": "info",
    "skipped": "muted",
    "failed": "danger",
    "removed": "muted",
}


def render(app: Any) -> None:
    st.markdown("## Knowledge Base")
    st.caption(
        "Load the synthetic corpus or upload your own documents, then inspect exactly what was "
        "parsed, what was indexed and what failed."
    )

    inventory = app.inventory()
    index = inventory["index"]

    # -- health tiles -----------------------------------------------------
    tiles(
        [
            {"label": "Documents", "value": f"{inventory['document_count']}"},
            {"label": "Indexed", "value": f"{inventory['indexed_count']}", "tone": "good"},
            {"label": "Passages", "value": f"{inventory['chunk_count']:,}", "tone": "accent"},
            {
                "label": "Failed",
                "value": f"{inventory['failed_count']}",
                "tone": "warn" if inventory["failed_count"] else "",
            },
            {
                "label": "Index status",
                "value": "Ready" if index.get("loaded") else "Not built",
                "detail": f"{index.get('backend', 'n/a')} · {index.get('vectors', 0):,} vectors",
            },
        ]
    )

    if index.get("stale"):
        banner(
            "warn",
            "The vector index and the metadata store disagree on passage count. Rebuild the index "
            "to bring them back into step.",
        )
    if index.get("model_matches") is False:
        banner(
            "danger",
            "The saved index was built with a different embedding model or chunking configuration. "
            f"{esc(index.get('mismatch_reason', ''))} Rebuild the index before relying on results.",
        )

    st.divider()

    # -- load corpus ------------------------------------------------------
    st.markdown("### Load the synthetic corpus")
    st.caption(
        "The bundled corpus is generated from a fixed seed, so it is identical on every machine. "
        "It contains 24 fictional candidate profiles, 7 job descriptions and 7 hiring documents "
        "across CSV, JSON, Markdown, TXT, PDF and DOCX."
    )

    columns = st.columns([1, 1, 1])
    with columns[0]:
        if st.button("Load / refresh corpus", type="primary", use_container_width=True):
            _load_corpus(app, force=False)
    with columns[1]:
        if st.button("Force re-embed everything", use_container_width=True):
            _load_corpus(app, force=True)
    with columns[2]:
        if st.button("Rebuild index", use_container_width=True):
            _rebuild(app)

    # -- upload -----------------------------------------------------------
    st.divider()
    st.markdown("### Upload documents")
    st.caption(
        f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}. "
        f"Maximum {app.settings.max_upload_mb} MB per file. Scanned PDFs need OCR, which this MVP "
        "does not perform."
    )

    if app.settings.external_calls_enabled:
        banner(
            "warn",
            "<strong>External provider active.</strong> The configured embedding or generation "
            "model sends document and query text to an external API. Do not upload confidential "
            "material while an external provider is selected — see docs/PRIVACY_AND_RESPONSIBLE_USE.md.",
        )
    else:
        banner(
            "ok",
            "<strong>Runs locally.</strong> Embeddings and generation happen on this machine. "
            "No document text leaves it.",
        )

    uploads = st.file_uploader(
        "Drop files here",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
        key="kb_uploads",
    )

    collection = st.selectbox(
        "Assign uploads to a collection",
        options=["uploads", "candidate_profiles", "job_descriptions", "hiring_documents", "general"],
        format_func=lambda key: COLLECTIONS.get(key, key),
        key="kb_collection",
    )

    if uploads and st.button(f"Ingest {len(uploads)} file(s)", type="primary"):
        _ingest_uploads(app, uploads, collection)

    # -- inventory --------------------------------------------------------
    st.divider()
    st.markdown("### Document inventory")

    if not inventory["documents"]:
        empty_state(
            "No documents yet",
            "Load the synthetic corpus above, or upload your own files to build the knowledge base.",
        )
        return

    _inventory_charts(inventory)

    filter_columns = st.columns([1, 1, 1])
    with filter_columns[0]:
        status_filter = st.selectbox(
            "Status", ["all", "indexed", "failed", "skipped", "pending"], key="kb_status_filter"
        )
    with filter_columns[1]:
        collection_filter = st.selectbox(
            "Collection",
            ["all"] + sorted({d.collection for d in inventory["documents"]}),
            key="kb_collection_filter",
        )
    with filter_columns[2]:
        st.write("")
        st.write("")
        show_warnings_only = st.checkbox("Only rows with warnings", key="kb_warn_only")

    rows = [
        d
        for d in inventory["documents"]
        if (status_filter == "all" or d.status == status_filter)
        and (collection_filter == "all" or d.collection == collection_filter)
        and (not show_warnings_only or d.warnings or d.error)
    ]

    if not rows:
        st.info("No documents match these filters.")
        return

    for document in rows:
        _document_row(app, document)

    # -- issues -----------------------------------------------------------
    if inventory["failures"]:
        st.divider()
        st.markdown("### Parse failures")
        for failure in inventory["failures"]:
            st.markdown(
                f'<div class="err"><div class="t">{esc(failure["filename"])}</div>'
                f'<div class="m">{esc(failure["error"] or "Unknown error")}</div></div>',
                unsafe_allow_html=True,
            )

    if inventory["injection_flags"]:
        st.divider()
        st.markdown("### Prompt-injection flags")
        banner(
            "warn",
            "These documents contain text that reads like an instruction to the assistant. "
            "The content was treated as data, neutralised before reaching any model, and never "
            "executed. Review the file if you did not expect this.",
        )
        for flag in inventory["injection_flags"]:
            with st.expander(flag["filename"]):
                for warning in flag["warnings"]:
                    st.markdown(f"- {esc(warning)}")

    # -- export -----------------------------------------------------------
    st.divider()
    section_label("Export")
    download_row(
        [
            {
                "label": "Inventory (.csv)",
                "data": _inventory_csv(inventory),
                "filename": "knowledge_base_inventory.csv",
                "mime": "text/csv",
            },
            {
                "label": "Inventory (.md)",
                "data": _inventory_markdown(inventory),
                "filename": "knowledge_base_inventory.md",
                "mime": "text/markdown",
            },
        ]
    )


# --------------------------------------------------------------------------


def _load_corpus(app: Any, *, force: bool) -> None:
    progress = st.progress(0.0, text="Preparing corpus…")

    def update(fraction: float, message: str) -> None:
        progress.progress(min(1.0, max(0.0, fraction)), text=message)

    try:
        report = app.load_synthetic_corpus(force=force, progress=update)
    except HiringAssistantError as error:
        progress.empty()
        error_panel(error)
        return
    except Exception as error:  # pragma: no cover
        progress.empty()
        error_panel(error)
        return

    progress.empty()
    app.load_index()

    if report.failed:
        banner("warn", f"{len(report.failed)} file(s) could not be ingested. See the inventory below.")
    else:
        banner(
            "ok",
            f"Corpus ready — {report.summary} in {report.duration_seconds:.1f}s "
            f"({report.total_chunks_indexed:,} new passages embedded).",
        )

    st.rerun()


def _rebuild(app: Any) -> None:
    progress = st.progress(0.0, text="Rebuilding index…")

    def update(fraction: float, message: str) -> None:
        progress.progress(min(1.0, max(0.0, fraction)), text=message)

    try:
        report = app.rebuild_index(progress=update)
    except HiringAssistantError as error:
        progress.empty()
        error_panel(error)
        return
    except Exception as error:  # pragma: no cover
        progress.empty()
        error_panel(error)
        return

    progress.empty()
    banner("ok", f"Index rebuilt — {report.total_chunks_indexed:,} passages re-embedded.")
    st.rerun()


def _ingest_uploads(app: Any, uploads: list[Any], collection: str) -> None:
    files = [(upload.name, upload.getvalue()) for upload in uploads]
    progress = st.progress(0.0, text="Ingesting…")

    def update(fraction: float, message: str) -> None:
        progress.progress(min(1.0, max(0.0, fraction)), text=message)

    try:
        report = app.ingest_uploads(files, collection=collection, progress=update)
    except HiringAssistantError as error:
        progress.empty()
        error_panel(error)
        return
    except Exception as error:  # pragma: no cover
        progress.empty()
        error_panel(error)
        return

    progress.empty()
    app.load_index()

    for item in report.items:
        if item.status == "failed":
            banner("danger", f"<strong>{esc(item.filename)}</strong> — {esc(item.message)} {esc(item.remedy)}")
        elif item.status == "skipped":
            banner("info", f"<strong>{esc(item.filename)}</strong> — {esc(item.message)}")

    if report.indexed:
        banner("ok", f"Ingested {report.summary}.")
    st.rerun()


def _document_row(app: Any, document: Any) -> None:
    tone = STATUS_TONE.get(document.status, "muted")
    size = ""
    badge = pill(document.status.title(), tone)
    origin = pill("Synthetic" if document.synthetic else "Uploaded", "synthetic" if document.synthetic else "upload")

    with st.container(border=True):
        head = st.columns([3.2, 1.1, 1.1, 1.4])
        with head[0]:
            st.markdown(f"**{esc(document.filename)}**", unsafe_allow_html=True)
            st.markdown(
                f'<span class="caption">{esc(COLLECTIONS.get(document.collection, document.collection))} · '
                f"{document.chunk_count} passage(s) · {document.char_count:,} characters</span>",
                unsafe_allow_html=True,
            )
        with head[1]:
            st.markdown(f"{badge} {origin}", unsafe_allow_html=True)
        with head[2]:
            if document.indexed_at:
                st.markdown(
                    f'<span class="caption">Indexed<br>{esc(document.indexed_at[:10])}</span>',
                    unsafe_allow_html=True,
                )
        with head[3]:
            actions = st.columns(2)
            with actions[0]:
                if st.button("Re-index", key=f"reidx_{document.document_id}", use_container_width=True):
                    _reindex(app, document.document_id)
            with actions[1]:
                if st.button("Remove", key=f"rm_{document.document_id}", use_container_width=True):
                    _remove(app, document.document_id)

        if document.warnings:
            with st.expander(f"{len(document.warnings)} parsing note(s)"):
                for warning in document.warnings:
                    st.markdown(f"- {esc(warning)}")

        if document.error:
            st.markdown(
                f'<div class="err"><div class="t">Ingestion failed</div>'
                f'<div class="m">{esc(document.error)}</div></div>',
                unsafe_allow_html=True,
            )

        if document.status == "indexed":
            with st.expander("Preview extracted text"):
                chunks = app.store.chunks_for_document(document.document_id)
                if chunks:
                    st.markdown(
                        f'<div class="cit-quote">{esc(chunks[0].text[:900])}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Showing passage 1 of {len(chunks)}. Each passage carries its page or "
                        "section so citations can point back to it."
                    )
                else:
                    st.caption("No passages stored for this document.")


def _reindex(app: Any, document_id: str) -> None:
    with st.spinner("Re-embedding…"):
        try:
            result = app.reindex_document(document_id)
        except HiringAssistantError as error:
            error_panel(error)
            return
        except Exception as error:  # pragma: no cover
            error_panel(error)
            return
    if result.status == "indexed":
        st.toast(result.message)
    else:
        error_panel(HiringAssistantError(result.message, remedy=result.remedy))
    st.rerun()


def _remove(app: Any, document_id: str) -> None:
    try:
        outcome = app.remove_document(document_id)
    except HiringAssistantError as error:
        error_panel(error)
        return
    except Exception as error:  # pragma: no cover
        error_panel(error)
        return
    st.toast(
        f"Removed {outcome['chunks_deleted']} passage(s) and {outcome['vectors_removed']} vector(s). "
        "The document can no longer appear in retrieval."
    )
    app.load_index()
    st.rerun()


# --------------------------------------------------------------------------


def _inventory_charts(inventory: dict[str, Any]) -> None:
    by_collection = inventory["chunks_by_collection"]
    if not by_collection:
        return

    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    labels = [COLLECTIONS.get(key, key).split(" (")[0] for key in by_collection]
    values = list(by_collection.values())
    order = sorted(range(len(values)), key=lambda i: -values[i])

    figure = go.Figure(
        go.Bar(
            x=[values[i] for i in order],
            y=[labels[i] for i in order],
            orientation="h",
            marker={"color": [SEQUENTIAL_BLUE[min(len(SEQUENTIAL_BLUE) - 1, 3 + (i % 3))] for i in order]},
            text=[f"{values[i]:,}" for i in order],
            textposition="outside",
            textfont={"size": 11, "color": COLORS["slate_500"]},
            hovertemplate="%{y}: %{x} passages<extra></extra>",
        )
    )
    layout = plotly_layout(height=max(200, 44 * len(values) + 60))
    layout["margin"] = {"l": 8, "r": 44, "t": 30, "b": 8}
    layout["yaxis"] = {"automargin": True, "autorange": "reversed", "tickfont": {"size": 12}}
    layout["xaxis"] = {
        "showgrid": True,
        "gridcolor": COLORS["slate_100"],
        "tickfont": {"size": 11, "color": COLORS["slate_500"]},
    }
    layout["title"] = {
        "text": "Indexed passages by collection",
        "font": {"size": 13, "color": COLORS["navy_900"]},
    }
    figure.update_layout(**layout)
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def _inventory_csv(inventory: dict[str, Any]) -> str:
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "document_id", "filename", "file_type", "collection", "status", "origin",
            "chunk_count", "char_count", "page_count", "synthetic", "indexed_at",
            "warning_count", "error",
        ]
    )
    for document in inventory["documents"]:
        writer.writerow(
            [
                document.document_id,
                document.filename,
                document.file_type,
                document.collection,
                document.status,
                document.origin,
                document.chunk_count,
                document.char_count,
                document.page_count if document.page_count is not None else "",
                document.synthetic,
                document.indexed_at or "",
                len(document.warnings),
                document.error or "",
            ]
        )
    return buffer.getvalue()


def _inventory_markdown(inventory: dict[str, Any]) -> str:
    lines = [
        "# Knowledge Base Inventory",
        "",
        f"- Documents: {inventory['document_count']}",
        f"- Indexed: {inventory['indexed_count']}",
        f"- Passages: {inventory['chunk_count']:,}",
        f"- Failed: {inventory['failed_count']}",
        "",
        "| Filename | Collection | Status | Passages | Characters | Origin | Warnings |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for document in inventory["documents"]:
        lines.append(
            f"| {document.filename} | {document.collection} | {document.status} | "
            f"{document.chunk_count} | {document.char_count:,} | {document.origin} | "
            f"{len(document.warnings)} |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = ["render"]
