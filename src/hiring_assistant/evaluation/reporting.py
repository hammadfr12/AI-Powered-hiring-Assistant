"""Markdown and CSV rendering of evaluation results."""

from __future__ import annotations

import csv
import io
from typing import Any

from ..models import EvaluationReport

METRIC_LABELS: dict[str, str] = {
    "cases": "Cases run",
    "passed": "Cases passed",
    "pass_rate": "Pass rate",
    "hit_rate": "Hit rate (expected source retrieved)",
    "recall_at_k": "Recall@k",
    "precision_at_k": "Precision@k",
    "citation_validity": "Citation validity",
    "abstention_accuracy": "Abstention accuracy",
    "injection_resistance": "Injection resistance",
    "answered_with_citations": "Answered with citations",
    "latency_p50_ms": "Latency p50 (ms)",
    "latency_p95_ms": "Latency p95 (ms)",
    "mean_latency_ms": "Mean latency (ms)",
    "error_rate": "Error rate",
}

#: Metrics rendered as percentages rather than raw decimals.
PERCENT_METRICS = {
    "pass_rate", "hit_rate", "recall_at_k", "precision_at_k", "citation_validity",
    "abstention_accuracy", "injection_resistance", "answered_with_citations", "error_rate",
}


def format_metric(key: str, value: float) -> str:
    if key in PERCENT_METRICS:
        return f"{value * 100:.1f}%"
    if key.endswith("_ms"):
        return f"{value:,.1f}"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.3f}"


def report_to_markdown(report: EvaluationReport) -> str:
    """Render a full evaluation report as Markdown."""
    lines: list[str] = []
    config = report.config or {}

    lines.append("# RAG Evaluation Report")
    lines.append("")
    lines.append(f"- **Run id:** `{report.run_id}`")
    lines.append(f"- **Generated:** {report.created_at}")
    lines.append(f"- **Duration:** {report.duration_seconds:.2f} s")
    lines.append(f"- **Embedding model:** `{config.get('embedding_model', 'n/a')}`")
    lines.append(f"- **Generation model:** `{config.get('generation_model', 'n/a')}`")
    lines.append(
        f"- **Retrieval:** top-k={config.get('top_k')}, "
        f"min relevance={config.get('min_relevance')}"
    )
    lines.append(
        f"- **Chunking:** size={config.get('chunk_size')}, overlap={config.get('chunk_overlap')}"
    )
    lines.append("")

    lines.append("## Summary metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    for key, label in METRIC_LABELS.items():
        if key in report.metrics:
            lines.append(f"| {label} | {format_metric(key, report.metrics[key])} |")
    lines.append("")

    grouped: dict[str, list[Any]] = {}
    for result in report.results:
        grouped.setdefault(result.category, []).append(result)

    lines.append("## Results by category")
    lines.append("")
    lines.append("| Category | Cases | Passed | Pass rate | Hit rate | Mean recall | Mean latency (ms) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for category, items in sorted(grouped.items()):
        passed = sum(1 for r in items if r.passed)
        hit = sum(1 for r in items if r.hit)
        recall = sum(r.recall for r in items) / len(items)
        latency = sum(r.latency_ms for r in items) / len(items)
        lines.append(
            f"| {category} | {len(items)} | {passed} | "
            f"{passed / len(items) * 100:.0f}% | {hit / len(items) * 100:.0f}% | "
            f"{recall:.2f} | {latency:,.0f} |"
        )
    lines.append("")

    lines.append("## Case detail")
    lines.append("")
    lines.append("| Case | Category | Status | Hit | Citations | Abstained | Blocked | Latency (ms) | Notes |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        notes = (result.failure_reason or "").replace("|", "/")[:110]
        lines.append(
            f"| `{result.case_id}` | {result.category} | {status} | "
            f"{'yes' if result.hit else 'no'} | {result.citation_count} | "
            f"{'yes' if result.abstained else 'no'} | {'yes' if result.blocked else 'no'} | "
            f"{result.latency_ms:,.0f} | {notes} |"
        )
    lines.append("")

    if report.failures:
        lines.append("## Failures")
        lines.append("")
        for failure in report.failures:
            lines.append(f"- {failure}")
        lines.append("")
    else:
        lines.append("## Failures")
        lines.append("")
        lines.append("None. Every case met its expectation in this run.")
        lines.append("")

    lines.append("## How to read these numbers")
    lines.append("")
    lines.append(
        "- **Hit rate / Recall@k** compare retrieved *filenames* against the sources each case "
        "was written to expect. They measure retrieval, not answer quality."
    )
    lines.append(
        "- **Citation validity** is a hard correctness check: it is 100% only when every "
        "rendered citation resolved to a chunk that was retrieved for that exact query."
    )
    lines.append(
        "- **Abstention accuracy** counts cases where refusing to answer was the correct "
        "behaviour. Abstaining is a success, not a failure."
    )
    lines.append(
        "- **Injection resistance** counts hostile cases that were correctly refused. The "
        "document-side injection case is expected to *answer* while refusing the embedded "
        "instruction, so it is scored on content, not refusal."
    )
    lines.append(
        "- These figures come from a synthetic corpus with a fixed seed. They demonstrate that "
        "the pipeline behaves as specified; they are not a claim about performance on real "
        "recruitment data."
    )
    lines.append("")

    return "\n".join(lines)


def report_to_csv(report: EvaluationReport) -> str:
    """Per-case CSV export."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "case_id", "category", "query", "passed", "hit", "recall", "precision_at_k",
            "citation_valid", "citation_count", "abstained", "abstention_correct",
            "blocked", "blocked_correct", "latency_ms", "retrieved_sources", "failure_reason",
        ]
    )
    for result in report.results:
        writer.writerow(
            [
                result.case_id,
                result.category,
                result.query,
                result.passed,
                result.hit,
                result.recall,
                result.precision_at_k,
                result.citation_valid,
                result.citation_count,
                result.abstained,
                result.abstention_correct,
                result.blocked,
                result.blocked_correct,
                f"{result.latency_ms:.2f}",
                "; ".join(result.retrieved_source_ids),
                result.failure_reason or "",
            ]
        )
    return buffer.getvalue()


def metrics_table_rows(report: EvaluationReport) -> list[dict[str, str]]:
    """Rows shaped for a Streamlit dataframe."""
    rows: list[dict[str, str]] = []
    for key, label in METRIC_LABELS.items():
        if key in report.metrics:
            rows.append({"Metric": label, "Value": format_metric(key, report.metrics[key])})
    return rows


def case_table_rows(report: EvaluationReport) -> list[dict[str, Any]]:
    """Rows shaped for a Streamlit dataframe."""
    rows: list[dict[str, Any]] = []
    for result in report.results:
        rows.append(
            {
                "Case": result.case_id,
                "Category": result.category,
                "Status": "PASS" if result.passed else "FAIL",
                "Hit": "yes" if result.hit else "no",
                "Recall": round(result.recall, 2),
                "Citations": result.citation_count,
                "Abstained": "yes" if result.abstained else "no",
                "Blocked": "yes" if result.blocked else "no",
                "Latency (ms)": round(result.latency_ms, 1),
                "Note": (result.failure_reason or "")[:120],
            }
        )
    return rows


__all__ = [
    "METRIC_LABELS",
    "case_table_rows",
    "format_metric",
    "metrics_table_rows",
    "report_to_csv",
    "report_to_markdown",
]
