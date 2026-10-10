"""AI-Powered Hiring Assistant — Streamlit entry point.

Run with:

    streamlit run app.py

The app is a retrieval assistant: it searches and summarises the documents you
supply, cites the exact passage behind every statement, and abstains when the
corpus does not contain the answer.  It does not score, rank or select
candidates.

This module owns the page shell only.  All behaviour lives in
``src/hiring_assistant`` so that the pipeline is testable without Streamlit.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src/` importable when running `streamlit run app.py` from the repo root.
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st  # noqa: E402

from hiring_assistant import __version__  # noqa: E402
from hiring_assistant.config import load_settings  # noqa: E402
from hiring_assistant.errors import HiringAssistantError  # noqa: E402
from hiring_assistant.services import HiringAssistant  # noqa: E402
from hiring_assistant.ui import components as ui  # noqa: E402
from hiring_assistant.ui.sidebar import render_sidebar  # noqa: E402
from hiring_assistant.ui.theme import build_css  # noqa: E402
from hiring_assistant.ui.views import ask, evaluation, explorer, knowledge, search  # noqa: E402

PAGE_TITLE = "AI-Powered Hiring Assistant"
PAGE_ICON = "🧭"

VIEWS = {
    "Ask Assistant": ask.render,
    "Semantic Search": search.render,
    "Candidate Explorer": explorer.render,
    "Knowledge Base": knowledge.render,
    "Evaluation & Settings": evaluation.render,
}


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Starting the assistant…")
def get_app(config_signature: str) -> HiringAssistant:
    """Build the service once per Streamlit process.

    Keyed on a signature of the configuration so that changing models in
    ``.env`` and restarting produces a fresh instance rather than reusing one
    built with the old embedding model.
    """
    del config_signature  # only used as a cache key
    return HiringAssistant(load_index=True)


def _ensure_corpus(app: HiringAssistant) -> None:
    """On a genuinely fresh install, build the corpus and index automatically.

    This is what makes "clone and run" work with no manual step: if nothing has
    ever been indexed, the bundled synthetic corpus is loaded once.  It is
    guarded on an empty store so it never re-embeds on later runs.
    """
    if app.store.count_documents() > 0:
        return

    with st.spinner("First run — building the synthetic corpus and index…"):
        try:
            app.load_synthetic_corpus()
            app.load_index()
        except HiringAssistantError as error:
            ui.error_panel(error)
        except Exception as error:  # pragma: no cover - defensive
            ui.error_panel(error)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(build_css(), unsafe_allow_html=True)

    settings = load_settings()
    signature = "|".join(
        [
            settings.embedding_model,
            settings.generation_model,
            str(settings.chunk_size),
            str(settings.chunk_overlap),
            str(settings.db_path),
            str(settings.index_dir),
            __version__,
        ]
    )

    try:
        app = get_app(signature)
    except Exception as error:  # pragma: no cover - startup failure
        st.error("The assistant could not start.")
        ui.error_panel(error)
        st.stop()
        return

    if "session_id" not in st.session_state:
        st.session_state.session_id = "sess_default"
        st.session_state._run = 0

    _ensure_corpus(app)

    ui.app_header(
        PAGE_TITLE,
        "Evidence-grounded search and Q&A across candidate, role and hiring documents",
        app.generator.mode_label,
        app.settings.external_calls_enabled,
    )

    render_sidebar(app)

    # Warnings collected during start-up (fallbacks, stale index, mismatches).
    for warning in dict.fromkeys(app.warnings):
        ui.banner("warn", ui.esc(warning))

    ui.synthetic_notice()

    names = list(VIEWS)
    view = st.radio(
        "View",
        options=names,
        horizontal=True,
        label_visibility="collapsed",
        key="active_view",
    )

    try:
        VIEWS[view](app)
    except HiringAssistantError as error:
        ui.error_panel(error)
    except Exception as error:  # pragma: no cover - surface, never white-screen
        ui.error_panel(error)
        with st.expander("Technical detail"):
            import traceback

            st.code(traceback.format_exc(), language="text")

    st.divider()
    st.markdown(
        f'<div class="caption">AI-Powered Hiring Assistant v{__version__} · '
        "retrieval and summarisation only — no hiring decisions · "
        f"embedding: <code>{ui.esc(app.settings.embedding_model_id)}</code> · "
        f"generation: <code>{ui.esc(app.settings.generation_model_id)}</code></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
