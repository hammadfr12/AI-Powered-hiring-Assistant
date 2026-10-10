#!/usr/bin/env python
"""Build or rebuild the vector index from the documents already in the store.

Usage:
    python scripts/build_index.py [--load-corpus] [--rebuild] [--stats]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hiring_assistant.services import HiringAssistant  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the FAISS index.")
    parser.add_argument("--load-corpus", action="store_true", help="Ingest the synthetic corpus first.")
    parser.add_argument("--rebuild", action="store_true", help="Re-embed every stored chunk.")
    parser.add_argument("--force-corpus", action="store_true", help="Re-ingest even unchanged files.")
    parser.add_argument("--stats", action="store_true", help="Print index health afterwards.")
    args = parser.parse_args()

    app = HiringAssistant()

    if args.load_corpus:
        report = app.load_synthetic_corpus(force=args.force_corpus)
        print(f"Corpus: {report.summary}")
        for item in report.failed:
            print(f"  FAILED  {item.filename}: {item.message}")
        app.load_index()

    if args.rebuild:
        report = app.rebuild_index()
        print(f"Rebuilt: {report.total_chunks_indexed:,} passages re-embedded.")
    elif not args.load_corpus:
        app.load_index()

    health = app.index_health()
    print()
    print("Index health")
    print(f"  backend            : {health.get('backend')}")
    print(f"  loaded             : {health.get('loaded')}")
    print(f"  vectors            : {health.get('vectors', 0):,}")
    print(f"  dimension          : {health.get('dimension')}")
    print(f"  chunks in store    : {health.get('chunks_in_store', 0):,}")
    print(f"  indexed documents  : {health.get('indexed_documents', 0)}")
    print(f"  embedding model    : {health.get('embedding_model_id', 'n/a')}")
    print(f"  matches config     : {health.get('matches_configuration', 'n/a')}")
    if health.get("stale"):
        print("  WARNING: index and store disagree on passage count — run with --rebuild.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
