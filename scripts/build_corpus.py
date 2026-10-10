#!/usr/bin/env python
"""Generate the deterministic synthetic corpus on disk.

Usage:
    python scripts/build_corpus.py [--force] [--candidates N] [--seed N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hiring_assistant.synthetic import (  # noqa: E402
    DEFAULT_SEED,
    corpus_is_stale,
    write_corpus,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic demo corpus.")
    parser.add_argument("--force", action="store_true", help="Regenerate even if the corpus looks current.")
    parser.add_argument("--candidates", type=int, default=24, help="Number of candidate profiles (default 24).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "synthetic",
        help="Target directory.",
    )
    args = parser.parse_args()

    if not args.force and not corpus_is_stale(args.output, candidate_count=args.candidates, seed=args.seed):
        print(f"Corpus at {args.output} is current. Use --force to regenerate.")
        return 0

    corpus = write_corpus(args.output, candidate_count=args.candidates, seed=args.seed)

    print(f"Wrote corpus to {args.output}")
    print(f"  candidate profiles : {corpus.candidate_count}")
    print(f"  job descriptions   : {len(corpus.jobs)}")
    print(f"  hiring documents   : {len(corpus.documents)}")
    print(f"  corpus files       : {len(corpus.files)}")
    print(f"  malformed fixtures : {len(corpus.malformed)}")
    print(f"  seed               : {corpus.seed}")
    print()
    print("All content is fictional and machine-generated. It contains no real personal data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
