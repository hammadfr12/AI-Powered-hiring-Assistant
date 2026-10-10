#!/usr/bin/env python
"""Run the deterministic RAG benchmark and write Markdown + CSV reports.

Usage:
    python scripts/run_evaluation.py [--top-k N] [--min-relevance F] [--output DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hiring_assistant.evaluation.reporting import (  # noqa: E402
    METRIC_LABELS,
    format_metric,
    report_to_csv,
    report_to_markdown,
)
from hiring_assistant.evaluation.runner import RunnerConfig  # noqa: E402
from hiring_assistant.services import HiringAssistant  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation benchmark.")
    parser.add_argument("--top-k", type=int, default=None, help="Passages retrieved per query.")
    parser.add_argument("--min-relevance", type=float, default=None, help="Abstention threshold.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs",
        help="Directory for the generated reports.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the summary table.")
    args = parser.parse_args()

    app = HiringAssistant()

    if not app.index_loaded:
        print("No index found — loading the synthetic corpus first.")
        report = app.load_synthetic_corpus()
        print(f"  {report.summary}")
        app.load_index()

    if not app.index_loaded:
        print("ERROR: could not build an index. Run scripts/build_index.py --load-corpus.")
        return 1

    config = RunnerConfig(
        top_k=args.top_k or app.settings.top_k,
        min_relevance=(
            args.min_relevance if args.min_relevance is not None else app.settings.min_relevance
        ),
    )

    def progress(fraction: float, message: str) -> None:
        if not args.quiet:
            print(f"  [{fraction * 100:5.1f}%] {message}")

    print(f"Running {config.top_k=} {config.min_relevance=}".replace("config.", ""))
    report = app.evaluate(config=config, progress=progress)

    print()
    print(f"Evaluation complete — {len(report.results)} cases in {report.duration_seconds:.2f}s")
    print()
    width = max(len(label) for label in METRIC_LABELS.values())
    for key, label in METRIC_LABELS.items():
        if key in report.metrics:
            print(f"  {label.ljust(width)}  {format_metric(key, report.metrics[key])}")

    if report.failures:
        print()
        print(f"Failures ({len(report.failures)}):")
        for failure in report.failures:
            print(f"  - {failure}")
    else:
        print()
        print("No failures — every case met its expectation.")

    args.output.mkdir(parents=True, exist_ok=True)
    markdown_path = args.output / "EVALUATION_RESULTS.md"
    csv_path = args.output / "EVALUATION_RESULTS.csv"
    markdown_path.write_text(report_to_markdown(report), encoding="utf-8")
    csv_path.write_text(report_to_csv(report), encoding="utf-8")

    print()
    print(f"Wrote {markdown_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
