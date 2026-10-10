#!/usr/bin/env python
"""Package the project as a distributable ZIP.

Excluded, deliberately:

* virtual environments, caches and build artefacts
* ``.env`` and any other secret-bearing file
* the regenerable vector index, SQLite database and generated corpus

The index and corpus are both rebuilt from a fixed seed by the documented
commands, so shipping them would only bloat the archive and risk shipping a
stale artefact.  ``README.md`` documents the two commands that recreate them.

Usage:
    python scripts/make_zip.py [--output PATH] [--verify]
"""

from __future__ import annotations

import argparse
import zipfile
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT.parent / "AI-Powered-Hiring-Assistant.zip"
ARCHIVE_ROOT = "AI-Powered-Hiring-Assistant"

#: Directory names skipped anywhere in the tree.
EXCLUDED_DIRS = {
    ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", "htmlcov", ".git", ".idea", ".vscode", "node_modules",
    "build", "dist", ".eggs", "_malformed",
    # Contains machine-specific absolute paths (a local preview server config),
    # so it is not meaningful on another machine.
    ".claude",
}

#: Glob patterns for files that are never included.
EXCLUDED_GLOBS = (
    "*.pyc", "*.pyo", "*.so", "*.dylib", "*.egg-info", ".DS_Store", "Thumbs.db",
    "*.sqlite3", "*.sqlite3-wal", "*.sqlite3-shm", "*.log", ".coverage*",
)

#: Files that must never be packaged, whatever else is included.
EXCLUDED_FILES = {".env", ".env.local", ".env.production", "secrets.json"}

#: Paths relative to the project root that are excluded even though their
#: directory would otherwise be included.
EXCLUDED_PATHS = {
    "data/index",
    "data/synthetic",
}


def _is_excluded(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT)
    parts = relative.parts

    if any(part in EXCLUDED_DIRS for part in parts):
        return True
    if path.name in EXCLUDED_FILES:
        return True
    if path.name.startswith(".env.") and path.name != ".env.example":
        return True
    for pattern in EXCLUDED_GLOBS:
        if path.match(pattern):
            return True

    relative_posix = relative.as_posix()
    for excluded in EXCLUDED_PATHS:
        if relative_posix == excluded or relative_posix.startswith(excluded + "/"):
            return True
    return False


def collect_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if _is_excluded(path):
            continue
        files.append(path)
    return files


def build_zip(output: Path) -> tuple[Path, list[str], int]:
    files = collect_files()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        output.unlink()

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            arcname = f"{ARCHIVE_ROOT}/{path.relative_to(PROJECT_ROOT).as_posix()}"
            archive.write(path, arcname)

        # Record how the archive was produced, inside the archive.
        stamp = datetime.now(UTC).replace(microsecond=0).isoformat()
        manifest = "\n".join(
            [
                f"AI-Powered Hiring Assistant — archive built {stamp}",
                "",
                f"Files included: {len(files)}",
                "",
                "Excluded by design:",
                "  - virtual environments, caches and build artefacts",
                "  - .env and any secret-bearing file",
                "  - data/index (vector index — rebuild with: make index)",
                "  - data/synthetic (demo corpus — rebuild with: make corpus)",
                "  - data/*.sqlite3 (metadata store — rebuilt on first run)",
                "",
                "To recreate everything from a clean checkout:",
                "  make install      # create .venv and install dependencies",
                "  make corpus       # regenerate the synthetic demo corpus",
                "  make index        # build the vector index",
                "  make run          # launch the app",
                "",
                "The app also builds the corpus and index automatically on first launch,",
                "so `make install && make run` is sufficient.",
                "",
            ]
        )
        archive.writestr(f"{ARCHIVE_ROOT}/ARCHIVE_CONTENTS.txt", manifest)

    names = [
        f"{ARCHIVE_ROOT}/{path.relative_to(PROJECT_ROOT).as_posix()}"
        for path in files
    ] + [f"{ARCHIVE_ROOT}/ARCHIVE_CONTENTS.txt"]
    return output, names, len(files)


def verify_zip(path: Path, expected: list[str]) -> list[str]:
    """Verify the archive opens, is non-empty and contains the key files."""
    problems: list[str] = []

    if not path.exists():
        return [f"archive does not exist: {path}"]
    if path.stat().st_size == 0:
        return [f"archive is empty: {path}"]

    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                problems.append(f"corrupt member: {bad}")
            names = set(archive.namelist())
    except zipfile.BadZipFile as exc:
        return [f"not a valid zip: {exc}"]

    required = {
        f"{ARCHIVE_ROOT}/app.py",
        f"{ARCHIVE_ROOT}/README.md",
        f"{ARCHIVE_ROOT}/requirements.txt",
        f"{ARCHIVE_ROOT}/Makefile",
        f"{ARCHIVE_ROOT}/.env.example",
        f"{ARCHIVE_ROOT}/.gitignore",
        f"{ARCHIVE_ROOT}/pyproject.toml",
        f"{ARCHIVE_ROOT}/src/hiring_assistant/config.py",
        f"{ARCHIVE_ROOT}/src/hiring_assistant/services.py",
        f"{ARCHIVE_ROOT}/tests/conftest.py",
        f"{ARCHIVE_ROOT}/docs/ARCHITECTURE.md",
        f"{ARCHIVE_ROOT}/docs/EVALUATION.md",
        f"{ARCHIVE_ROOT}/docs/PRIVACY_AND_RESPONSIBLE_USE.md",
        f"{ARCHIVE_ROOT}/docs/DATA_DICTIONARY.md",
    }
    for name in sorted(required - names):
        problems.append(f"missing expected file: {name}")

    leaked = [name for name in names if Path(name).name == ".env"]
    if leaked:
        problems.append(f"SECRET LEAK: archive contains {leaked}")

    missing = set(expected) - names
    if missing:
        problems.append(f"{len(missing)} collected file(s) missing from the archive")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Package the project as a ZIP archive.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Target .zip path.")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify after writing.")
    args = parser.parse_args()

    path, names, count = build_zip(args.output)

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Wrote {path}")
    print(f"  files : {count + 1} (including ARCHIVE_CONTENTS.txt)")
    print(f"  size  : {size_mb:.2f} MB")

    problems = verify_zip(path, names) if args.verify else []
    if problems:
        print()
        print("VERIFICATION FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print()
    print("Verification passed:")
    print("  archive opens and passes CRC checks")
    print(f"  all {len(names)} expected entries present")
    print("  no .env or secret files included")
    print("  index, database and generated corpus correctly excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
