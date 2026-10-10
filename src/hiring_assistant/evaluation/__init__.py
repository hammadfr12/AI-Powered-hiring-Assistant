"""Deterministic RAG evaluation: cases, runner and reporting."""

from .cases import CATEGORY_EXPECTATIONS, CaseSet, build_cases
from .reporting import (
    METRIC_LABELS,
    case_table_rows,
    format_metric,
    metrics_table_rows,
    report_to_csv,
    report_to_markdown,
)
from .runner import ABSTENTION_CATEGORIES, HOSTILE_CATEGORIES, EvaluationRunner, RunnerConfig

__all__ = [
    "ABSTENTION_CATEGORIES",
    "CATEGORY_EXPECTATIONS",
    "HOSTILE_CATEGORIES",
    "METRIC_LABELS",
    "CaseSet",
    "EvaluationRunner",
    "RunnerConfig",
    "build_cases",
    "case_table_rows",
    "format_metric",
    "metrics_table_rows",
    "report_to_csv",
    "report_to_markdown",
]
