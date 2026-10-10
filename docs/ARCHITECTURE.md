# Architecture

## Overview

The application is a retrieval-augmented generation (RAG) pipeline behind a
Streamlit workspace. It is layered so that the pipeline can be driven, tested
and evaluated without the UI.

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit UI          app.py, src/hiring_assistant/ui/     │
│  Ask · Search · Explorer · Knowledge Base · Evaluation      │
└────────────────────────────┬────────────────────────────────┘
                             │  one facade, no direct storage access
┌────────────────────────────▼────────────────────────────────┐
│  Service layer         services.HiringAssistant             │
│  session lifecycle, corpus loading, exports, diagnostics    │
└──┬──────────┬────────────┬──────────────┬───────────────────┘
   │          │            │              │
┌──▼───────┐ ┌▼──────────┐ ┌▼───────────┐ ┌▼─────────────────┐
│ Ingestion│ │ Retrieval │ │ Generation │ │ Evidence         │
│ pipeline │ │ search    │ │ generator  │ │ requirements     │
│ parsers  │ │ embeddings│ │ citations  │ │ profiles         │
│ chunking │ │ filters   │ │ prompts    │ │                  │
└──┬───────┘ └─┬─────────┘ └─┬──────────┘ └─┬────────────────┘
   │           │             │              │
┌──▼───────────▼─────────────▼──────────────▼─────────────────┐
│  Storage           SQLite metadata · FAISS vector index     │
└─────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  Safety            guardrails · injection scanning          │
│                    untrusted-content neutralisation         │
└─────────────────────────────────────────────────────────────┘
```

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `config.py` | Every tunable. Resolves defaults → `.env` → explicit overrides. Falls back to local models when credentials are missing, recording the fallback. |
| `errors.py` | Structured error hierarchy. Every user-caused failure carries a `message` **and** an actionable `remedy`. |
| `models.py` | Domain dataclasses mirroring the SQLite schema one-to-one. |
| `_compat.py` | Preloads PyTorch before FAISS. See "Native library ordering" below. |
| `synthetic.py` | Deterministic corpus generator. Pure function of `(seed, candidate_count)`. |
| `ingestion/parsers.py` | Format-specific extraction producing `ParsedSection`s that retain page/section provenance. |
| `ingestion/chunking.py` | Structure-aware chunking: paragraph → sentence packing with overlap. |
| `ingestion/pipeline.py` | Validation, hashing, chunking, batched embedding, atomic index persistence. |
| `retrieval/embeddings.py` | Provider abstraction: local sentence-transformers, OpenAI REST, deterministic hashing fallback. |
| `retrieval/search.py` | Hybrid vector + lexical ranking, metadata pre-filtering, deduplication, corpus-gap detection. |
| `generation/prompts.py` | System policy and untrusted-data envelope construction. |
| `generation/citations.py` | Parses, validates and rewrites citation markers. The core correctness guarantee. |
| `generation/generator.py` | Extractive and LLM backends behind one interface, sharing the same post-processing. |
| `evidence/requirements.py` | Requirement extraction and the requirement-to-evidence matrix. |
| `evidence/profiles.py` | Candidate/job discovery and cited profile summaries. |
| `evaluation/` | 25 deterministic cases, the runner, and Markdown/CSV reporting. |
| `safety/` | Injection detection, untrusted-text neutralisation, query guardrails. |
| `storage/database.py` | Thread-safe SQLite wrapper. |
| `storage/vector_store.py` | FAISS index with atomic rebuild and fingerprint validation. |
| `services.py` | The single seam between UI and pipeline. |

## Data flow

### Ingestion

```
file bytes
  → validate (extension, size, emptiness, binary check)
  → hash (content + embedding model + chunk config)   ─── unchanged? reuse, stop
  → parse (page/section-aware)
  → scan for prompt injection                        ─── flag, neutralise
  → chunk (paragraph → sentence packing, overlap)
  → embed (batched)
  → write chunks to SQLite + vectors to FAISS
  → save index atomically
  → mark document indexed
```

The hash includes the embedding model and chunking configuration, not just the
file content. Changing the chunk size therefore correctly forces re-embedding
rather than silently reusing vectors built under a different configuration.

### Query

```
question
  → guardrail screen            ─── refused/redirected? stop, explain why
  → embed query (same model as the index)
  → FAISS search (over-fetch ×6 for filtering headroom)
  → metadata pre-filter
  → hybrid score = 0.72·vector + 0.28·lexical overlap
  → deduplicate near-identical passages
  → relevance gate              ─── below threshold? abstain
  → corpus-gap check            ─── vocabulary absent? abstain
  → generate (extractive or LLM)
  → validate citations          ─── none survive? withhold, abstain
  → render with citation cards
```

## Key design decisions

### Citation validation is the correctness boundary

The model is not trusted. It writes `[S3]` markers referring to numbered source
blocks. After generation:

1. Every marker is parsed out.
2. Markers that do not resolve to a retrieved passage are **removed** from the
   text and replaced with a visible token.
3. Quoted spans are checked against the cited passage; a quote that cannot be
   located verbatim is not shown as a quote.
4. If no citation survives, the answer is **withheld** and the assistant
   abstains instead.

This is why `citation_validity` is 100% in the benchmark: it is structurally
impossible to render an unresolvable citation.

### Abstention uses vocabulary coverage, not a similarity threshold

Cosine similarity cannot separate "asked about something the corpus covers"
from "asked about something adjacent to it" — measured on this corpus, the
distributions overlap heavily (answerable queries score 0.49–0.78; absent-topic
queries score 0.22–0.57).

What does separate them is whether the question's **content words exist in the
corpus at all**. A question the corpus can answer uses the corpus's vocabulary;
a question about an absent topic ("salary band", "sabbatical leave", "quantum
computing") introduces words that appear nowhere. See
`retrieval/search.py::_coverage`.

Morphological folding matters here: a plain dictionary lookup would treat
"container" as unknown because the corpus writes "containers", and "profiles" as
unknown because it writes "profile" — turning ordinary inflection into a false
abstention. The variant test is a *prefix* relationship with a bounded length
difference, which correctly rejects `quantum`/`quantitative` while accepting
`report`/`reporting`.

### Atomic index rebuild

`VectorStore.save_atomic` writes to `index.new/`, reads it back with a
throwaway store to validate the vector count, and only then swaps
`index/ → index.prev` and `index.new/ → index/`. A crash at any point leaves
the previous working index untouched. This is the "build a new index before
replacing the last valid index" requirement.

### Native library ordering (macOS arm64)

`faiss-cpu` and PyTorch each bundle an OpenMP runtime. If FAISS is imported
*before* PyTorch, the process aborts with a bus error the first time the
encoder runs — a hard crash with no Python traceback. `_compat.py` preloads
PyTorch and is imported from `hiring_assistant/__init__.py`, so the ordering is
correct regardless of which module the caller imports first.

### Thread safety

Streamlit reruns the script block on different threads. `MetadataStore` keeps
one connection per thread via `threading.local`, with WAL journaling so readers
and a writer can coexist.

### Optional dependencies

`faiss` and `plotly` are imported defensively. Without FAISS, a NumPy
inner-product search with identical semantics takes over. Without Plotly, the
charts are skipped and the tabular data still renders. The active backend is
reported in the UI.

## Data model

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md). Seven tables:
`documents`, `chunks`, `conversations`, `messages`, `evaluation_cases`,
`evaluation_runs`, `index_state`.

The `chunks.index_id` column and the FAISS row position are kept in step by
`MetadataStore.set_index_ids`. Chunk ids are deterministic
(`{document_id}::c{index:04d}`), which is what makes the content-hash skip
optimisation safe.

## Testing strategy

| Area | Tests |
| --- | --- |
| Validation, parsing, chunking, metadata | `tests/test_ingestion.py` |
| Embeddings, FAISS, filtering, dedupe, corpus gap | `tests/test_retrieval.py` |
| Citation validation, abstention, guardrails, provider failures, memory | `tests/test_generation.py` |
| Requirement extraction, matrix wording, summaries, injection, benchmark | `tests/test_evidence.py` |
| Corpus determinism, configuration, storage | `tests/test_corpus_and_config.py` |

The suite runs offline using `HashingEmbeddingProvider` — a deterministic
lexical projection — so it needs no model download and no API key. The real
embedding path is exercised by `scripts/smoke_test.py` and the app itself.

## Extending the application

**A new file format** — add a parser returning `ParseResult` and register it in
`PARSERS`. Nothing else changes.

**A new embedding or generation provider** — implement `EmbeddingProvider` or
add a generator with a `generate()` method, then register it in the relevant
catalogue in `config.py`.

**A new evaluation category** — add cases in `evaluation/cases.py` with a new
`category` string, and add it to `CATEGORY_EXPECTATIONS` and the UI's
`CATEGORY_LABELS`.

**A new view** — add a module under `ui/views/` exposing `render(app)`, then
register it in the `VIEWS` dict in `app.py`.
