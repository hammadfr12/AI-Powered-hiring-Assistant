"""AI-Powered Hiring Assistant — Streamlit entry point.

Run with: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from hiring_assistant import __version__
from hiring_assistant.config import load_settings
from hiring_assistant.errors import HiringAssistantError
from hiring_assistant.services import HiringAssistant
from hiring_assistant.ui import components as ui
from hiring_assistant.ui.sidebar import render_sidebar
from hiring_assistant.ui.theme import build_css
from hiring_assistant.ui.views import ask, evaluation, explorer, knowledge, search

PAGE_TITLE = "AI-Powered Hiring Assistant"
PAGE_ICON = "🧑‍💼"

VIEWS = {
    "Ask Assistant": ask.render,
    "Semantic Search": search.render,
    "Candidate Explorer": explorer.render,
    "Knowledge Base": knowledge.render,
    "Evaluation & Settings": evaluation.render,
}


@st.cache_resource(show_spinner="Starting the assistant…")
def get_app(config_signature: str) -> HiringAssistant:
    """Create one assistant instance per configuration signature."""
    del config_signature
    return HiringAssistant(load_index=True)


def _ensure_corpus(app: HiringAssistant) -> None:
    """Seed the synthetic demo corpus and index on a fresh installation."""
    if app.store.count_documents() > 0:
        return
    with st.spinner("First run — building the synthetic corpus and index…"):
        try:
            app.load_synthetic_corpus()
            app.load_index()
        except HiringAssistantError as error:
            ui.error_panel(error)
        except Exception as error:  # defensive startup boundary
            ui.error_panel(error)


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide", initial_sidebar_state="expanded")
    st.markdown(build_css(), unsafe_allow_html=True)
    settings = load_settings()
    signature = "|".join([
        settings.embedding_model, settings.generation_model,
        str(settings.chunk_size), str(settings.chunk_overlap),
        str(settings.db_path), str(settings.index_dir), __version__,
    ])
    try:
        app = get_app(signature)
    except Exception as error:  # keep startup failures understandable in the UI
        st.error("The assistant could not start.")
        ui.error_panel(error)
        st.stop()
        return

    st.session_state.setdefault("session_id", "sess_default")
    _ensure_corpus(app)
    ui.app_header(PAGE_TITLE, "Evidence-grounded search and Q&A across candidate, role and hiring documents", app.generator.mode_label, app.settings.external_calls_enabled)
    render_sidebar(app)
    for warning in dict.fromkeys(app.warnings):
        ui.banner("warn", ui.esc(warning))
    ui.synthetic_notice()
    view = st.radio("View", options=list(VIEWS), horizontal=True, label_visibility="collapsed", key="active_view")
    try:
        VIEWS[view](app)
    except HiringAssistantError as error:
        ui.error_panel(error)
    except Exception as error:  # defensive UI boundary
        ui.error_panel(error)
        with st.expander("Technical detail"):
            import traceback
            st.code(traceback.format_exc(), language="text")
    st.divider()
    st.markdown(
        f'<div class="caption">AI-Powered Hiring Assistant v{__version__} · retrieval and summarisation only — no hiring decisions</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
