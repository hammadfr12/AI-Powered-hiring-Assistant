"""Shared sidebar: corpus status, retrieval tuning and filter scope."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ..config import COLLECTIONS
from .components import esc

#: Session keys the views read for their shared filter scope.
FILTER_COLLECTION_KEY = "filter_collections"
FILTER_TYPE_KEY = "filter_file_types"


def render_sidebar(app: Any) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="side-block"><div class="h">Knowledge base</div>',
            unsafe_allow_html=True,
        )
        inventory = app.inventory()
        index = inventory["index"]

        _row("Documents", f"{inventory['document_count']}")
        _row("Indexed", f"{inventory['indexed_count']}")
        _row("Passages", f"{inventory['chunk_count']:,}")
        _row("Index", "Ready" if index.get("loaded") else "Not built")
        _row("Backend", str(index.get("backend", "n/a")).upper())
        if index.get("failed"):
            _row("Failed", f"{inventory['failed_count']}")
        st.markdown("</div>", unsafe_allow_html=True)

        if not app.index_loaded:
            st.warning("No index yet. Load the corpus from the Knowledge Base tab.")

        # -- scope ---------------------------------------------------------
        st.markdown('<div class="side-block"><div class="h">Search scope</div>', unsafe_allow_html=True)
        available = app.search_service.available_filters()

        collection_options = sorted(available["collections"]) or []
        collection_labels = {
            key: COLLECTIONS.get(key, key).split(" (")[0] for key in collection_options
        }
        selected_collections = st.multiselect(
            "Collections",
            options=collection_options,
            format_func=lambda key: f"{collection_labels[key]} ({available['collections'][key]})",
            key=FILTER_COLLECTION_KEY,
            placeholder="All collections",
            label_visibility="collapsed",
        )

        type_options = sorted(available["file_types"]) or []
        selected_types = st.multiselect(
            "File types",
            options=type_options,
            format_func=lambda key: f"{key.upper()} ({available['file_types'][key]})",
            key=FILTER_TYPE_KEY,
            placeholder="All file types",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if selected_collections or selected_types:
            st.caption(
                "Filters apply to Ask Assistant and Semantic Search. The Candidate Explorer always "
                "scopes to the selected candidate's own document."
            )

        st.divider()

        # -- models --------------------------------------------------------
        st.markdown('<div class="side-block"><div class="h">Models</div>', unsafe_allow_html=True)
        _row("Embedding", app.settings.embedding_model_id.split("/")[-1])
        _row("Generation", "extractive" if not app.generator.llm else app.settings.generation_model_id)
        _row("External calls", "yes" if app.settings.external_calls_enabled else "no")
        st.markdown("</div>", unsafe_allow_html=True)

        if app.settings.external_calls_enabled:
            st.warning("Document text is sent to an external API.")
        else:
            st.caption("Runs fully locally — no document text leaves this machine.")

        st.divider()

        # -- quick actions -------------------------------------------------
        st.markdown('<div class="side-block"><div class="h">Quick actions</div>', unsafe_allow_html=True)
        if st.button("Load synthetic corpus", use_container_width=True, disabled=app.index_loaded):
            with st.spinner("Loading corpus…"):
                report = app.load_synthetic_corpus()
                app.load_index()
            st.success(report.summary)
            st.rerun()

        if st.button("Clear conversation", use_container_width=True):
            session_id = st.session_state.get("session_id")
            if session_id:
                removed = app.clear_history(session_id)
                st.session_state.pop("last_result", None)
                st.success(f"Cleared {removed} message(s).")
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.caption(
            "**Synthetic demo corpus.** All profiles, roles and policies are fictional. "
            "This assistant retrieves and summarises evidence; it does not score, rank or select "
            "candidates."
        )


def _row(key: str, value: str) -> None:
    st.markdown(
        f'<div class="side-row"><span class="k">{esc(key)}</span>'
        f'<span class="v">{esc(value)}</span></div>',
        unsafe_allow_html=True,
    )


__all__ = ["FILTER_COLLECTION_KEY", "FILTER_TYPE_KEY", "render_sidebar"]
