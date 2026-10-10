"""Format-specific text extraction.

Each parser returns a list of :class:`ParsedSection` objects.  Keeping page /
section structure explicit (rather than flattening to one string) is what lets
the chunker attach a real ``page`` or ``section`` to every passage, which in
turn is what makes citations traceable back to a location a human can open.

Supported: PDF (``pypdf``), DOCX (``python-docx``), TXT / Markdown, CSV and
JSON.  Every parser is defensive: a malformed file produces a warning and
whatever text could be recovered, and only a genuinely unreadable file raises
:class:`ParseError`.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import EmptyDocumentError, ParseError, UnsupportedFileTypeError
from ..utils import collapse, normalise_text

# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass
class ParsedSection:
    """A contiguous block of extracted text with its provenance."""

    text: str
    section: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseResult:
    """Everything a parser extracted from one file."""

    sections: list[ParsedSection] = field(default_factory=list)
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)
    file_type: str = ""

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections if s.text.strip())

    @property
    def char_count(self) -> int:
        return len(self.full_text)

    @property
    def is_empty(self) -> bool:
        return self.char_count < 5

    def add_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------


def parse_pdf(data: bytes, filename: str) -> ParseResult:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover
        raise ParseError(
            "pypdf is not installed, so PDF files cannot be read.",
            remedy="Install it with `pip install pypdf`, or convert the file to DOCX/TXT.",
        ) from exc

    result = ParseResult(file_type="pdf")
    try:
        reader = PdfReader(io.BytesIO(data))
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                raise ParseError(
                    f"'{filename}' is password-protected.",
                    remedy="Remove the password and upload it again.",
                ) from None
            result.add_warning("PDF was encrypted; opened with an empty password.")
    except ParseError:
        raise
    except PdfReadError as exc:
        raise ParseError(f"'{filename}' is not a readable PDF: {exc}") from exc
    except Exception as exc:
        raise ParseError(f"Could not open '{filename}' as a PDF: {exc}") from exc

    try:
        result.page_count = len(reader.pages)
    except Exception:  # pragma: no cover - malformed page tree
        result.page_count = None

    empty_pages = 0
    for number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # a single bad page must not kill the document
            result.add_warning(f"Page {number} could not be extracted: {exc}")
            empty_pages += 1
            continue

        text = normalise_text(raw)
        if not text.strip():
            empty_pages += 1
            continue
        result.sections.append(ParsedSection(text=text, section=f"Page {number}", page=number))

    if result.page_count and empty_pages:
        ratio = empty_pages / max(1, result.page_count)
        if ratio > 0.5:
            result.add_warning(
                f"{empty_pages} of {result.page_count} pages contained no extractable text. "
                "This usually means the PDF is scanned images; OCR is out of scope for this MVP."
            )

    return result


# --------------------------------------------------------------------------
# DOCX
# --------------------------------------------------------------------------

_HEADING_STYLE_RE = re.compile(r"^heading\s*(\d+)", re.IGNORECASE)


def parse_docx(data: bytes, filename: str) -> ParseResult:
    try:
        import docx
        from docx.opc.exceptions import PackageNotFoundError
    except ImportError as exc:  # pragma: no cover
        raise ParseError(
            "python-docx is not installed, so DOCX files cannot be read.",
            remedy="Install it with `pip install python-docx`.",
        ) from exc

    result = ParseResult(file_type="docx")
    try:
        document = docx.Document(io.BytesIO(data))
    except PackageNotFoundError as exc:
        raise ParseError(
            f"'{filename}' is not a valid .docx package (it may be a renamed .doc or .rtf).",
            remedy="Open it in Word and re-save as .docx, or export to PDF/TXT.",
        ) from exc
    except Exception as exc:
        raise ParseError(f"Could not read '{filename}': {exc}") from exc

    current_heading: str | None = None
    buffer: list[str] = []
    tables_seen = 0

    def flush() -> None:
        nonlocal buffer
        body = normalise_text("\n".join(buffer))
        if body.strip():
            result.sections.append(ParsedSection(text=body, section=current_heading))
        buffer = []

    for paragraph in document.paragraphs:
        text = paragraph.text or ""
        style_name = ""
        try:
            style_name = (paragraph.style.name or "") if paragraph.style else ""
        except Exception:  # pragma: no cover - broken style reference
            style_name = ""

        match = _HEADING_STYLE_RE.match(style_name)
        if match and text.strip():
            flush()
            current_heading = collapse(text)[:120]
            continue
        if text.strip():
            buffer.append(text)
    flush()

    # Tables carry a lot of signal in resumes and requirement sheets.
    for table in getattr(document, "tables", []):
        rows: list[str] = []
        for row in table.rows:
            cells = [collapse(cell.text) for cell in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            tables_seen += 1
            result.sections.append(
                ParsedSection(
                    text=normalise_text("\n".join(rows)),
                    section=f"Table {tables_seen}",
                    metadata={"kind": "table"},
                )
            )

    if tables_seen:
        result.add_warning(f"Extracted {tables_seen} table(s) as pipe-delimited text.")

    try:
        result.page_count = len(document.sections) or None
    except Exception:  # pragma: no cover
        result.page_count = None

    return result


# --------------------------------------------------------------------------
# Plain text / Markdown
# --------------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$", re.MULTILINE)


def parse_text(data: bytes, filename: str, *, file_type: str = "txt") -> ParseResult:
    result = ParseResult(file_type=file_type)
    text = _decode(data, result, filename)
    if not text.strip():
        return result

    if file_type in {"md", "markdown"}:
        # Split on Markdown headings so sections are meaningful.
        matches = list(_MD_HEADING_RE.finditer(text))
        if matches:
            preface = text[: matches[0].start()].strip()
            if preface:
                result.sections.append(ParsedSection(text=normalise_text(preface), section="Preface"))
            for i, match in enumerate(matches):
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                body = text[match.end() : end].strip()
                heading = collapse(match.group(2))[:120]
                if body:
                    result.sections.append(
                        ParsedSection(text=normalise_text(body), section=heading)
                    )
            return result

    # Fall back to blank-line-separated blocks.
    for block in re.split(r"\n\s*\n", text):
        block = normalise_text(block)
        if block.strip():
            result.sections.append(ParsedSection(text=block))
    return result


def _decode(data: bytes, result: ParseResult, filename: str) -> str:
    """Decode bytes with a small ladder of encodings and report substitutions."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
            if encoding not in ("utf-8", "utf-8-sig"):
                result.add_warning(f"Decoded '{filename}' as {encoding} (not UTF-8).")
            return text
        except UnicodeDecodeError:
            continue
    result.add_warning(f"'{filename}' contained undecodable bytes; replaced with U+FFFD.")
    return data.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------


def parse_csv(data: bytes, filename: str) -> ParseResult:
    """Render each CSV row as a readable ``Header: value`` block.

    A candidate CSV has one profile per row.  Rendering each row as key/value
    text — rather than as raw comma soup — is what makes the row retrievable
    and citable as a coherent passage.
    """
    result = ParseResult(file_type="csv")
    text = _decode(data, result, filename)
    if not text.strip():
        return result

    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    try:
        reader = csv.DictReader(io.StringIO(text), dialect=dialect)
        rows = list(reader)
    except csv.Error as exc:
        raise ParseError(f"Could not parse '{filename}' as CSV: {exc}") from exc

    if not rows:
        result.add_warning("CSV contained a header but no data rows.")
        return result

    headers = list(rows[0].keys())
    if any(h is None for h in headers):
        result.add_warning("CSV had more values than headers on some rows; extras were dropped.")

    for index, row in enumerate(rows, start=1):
        lines: list[str] = []
        for key in headers:
            if not key:
                continue
            value = row.get(key)
            if value is None or str(value).strip() == "":
                continue
            lines.append(f"{str(key).strip()}: {collapse(str(value))}")
        if lines:
            result.sections.append(
                ParsedSection(
                    text="\n".join(lines),
                    section=f"Row {index}",
                    metadata={"row_index": index},
                )
            )

    result.page_count = None
    return result


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------


def parse_json(data: bytes, filename: str) -> ParseResult:
    """Flatten JSON into per-record key/value passages."""
    result = ParseResult(file_type="json")
    text = _decode(data, result, filename)
    if not text.strip():
        return result

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(
            f"'{filename}' is not valid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}.",
            remedy="Validate the file with a JSON linter, then upload it again.",
        ) from exc

    records: list[tuple[str, Any]]
    if isinstance(payload, list):
        records = [(f"Record {i}", item) for i, item in enumerate(payload, start=1)]
    elif isinstance(payload, dict):
        # A dict with a single list value ({"candidates": [...]}) is common.
        list_values = [k for k, v in payload.items() if isinstance(v, list)]
        if len(list_values) == 1 and len(payload) <= 2:
            key = list_values[0]
            records = [(f"Record {i}", item) for i, item in enumerate(payload[key], start=1)]
        else:
            records = [(str(k), v) for k, v in payload.items()]
    else:
        result.add_warning("JSON top level was a scalar; stored as a single passage.")
        records = [("Document", payload)]

    for label, record in records:
        body = _flatten_json(record)
        if body.strip():
            result.sections.append(ParsedSection(text=normalise_text(body), section=label))

    return result


def _flatten_json(value: Any, prefix: str = "", depth: int = 0) -> str:
    """Render nested JSON as indented ``Key: value`` lines."""
    if depth > 6:
        return f"{prefix}: [nested content omitted]" if prefix else "[nested content omitted]"

    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            rendered = _flatten_json(item, label, depth + 1)
            if rendered.strip():
                lines.append(rendered)
        return "\n".join(lines)

    if isinstance(value, list):
        lines = []
        for index, item in enumerate(value):
            label = f"{prefix}[{index}]"
            rendered = _flatten_json(item, label, depth + 1)
            if rendered.strip():
                lines.append(rendered)
        return "\n".join(lines)

    if value is None:
        return ""
    if isinstance(value, bool):
        return f"{prefix}: {'yes' if value else 'no'}"
    return f"{prefix}: {collapse(str(value))}"


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

PARSERS: dict[str, Callable[[bytes, str], ParseResult]] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".txt": lambda data, name: parse_text(data, name, file_type="txt"),
    ".md": lambda data, name: parse_text(data, name, file_type="md"),
    ".markdown": lambda data, name: parse_text(data, name, file_type="md"),
    ".csv": parse_csv,
    ".json": parse_json,
}

SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(sorted(PARSERS))


def parse_document(
    data: bytes,
    filename: str,
    *,
    allowed_extensions: tuple[str, ...] | None = None,
) -> ParseResult:
    """Parse ``data`` according to ``filename``'s extension.

    Raises :class:`UnsupportedFileTypeError` for unknown extensions and
    :class:`EmptyDocumentError` when nothing extractable was found.
    """
    extension = Path(filename).suffix.lower()
    allowed = allowed_extensions or SUPPORTED_EXTENSIONS

    if extension not in allowed or extension not in PARSERS:
        raise UnsupportedFileTypeError(
            f"'{extension or filename}' is not a supported file type.",
            remedy=f"Supported types: {', '.join(sorted(set(allowed) & set(PARSERS)))}.",
            context={"filename": filename, "extension": extension},
        )

    if not data:
        raise EmptyDocumentError(
            f"'{filename}' is empty (0 bytes).",
            remedy="Upload a file that contains text.",
        )

    result = PARSERS[extension](data, filename)

    if result.is_empty:
        raise EmptyDocumentError(
            f"'{filename}' parsed successfully but contained no usable text "
            f"({result.char_count} characters extracted).",
            context={"warnings": result.warnings},
        )

    return result


__all__ = [
    "PARSERS",
    "SUPPORTED_EXTENSIONS",
    "ParseResult",
    "ParsedSection",
    "parse_csv",
    "parse_document",
    "parse_docx",
    "parse_json",
    "parse_pdf",
    "parse_text",
]
