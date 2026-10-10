"""AI-Powered Hiring Assistant — a recruiter-facing, evidence-grounded RAG app.

This is an information-retrieval assistant.  It searches and summarises the
documents you give it, cites the exact passages behind every statement and
abstains when the corpus does not contain the answer.  It does not score,
rank or select candidates.
"""

__version__ = "1.0.0"

# Preload native libraries in the correct order before anything imports FAISS.
# This is an import side effect, and it is deliberate: see _compat.py for why.
from . import _compat  # noqa: F401

__all__ = ["__version__"]
