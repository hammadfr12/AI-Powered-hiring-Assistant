# AI-Powered Hiring Assistant

A recruiter-facing retrieval-augmented generation (RAG) application for
natural-language search and evidence-grounded Q&A across candidate profiles, job
descriptions and approved hiring documents.

Every statement it makes carries a citation that resolves to a real retrieved
passage. When the documents do not contain the answer, it says so instead of
guessing.

> **This is an information retrieval assistant, not a hiring decision system.**
> It does not score, rank or select candidates, and it will not assess protected
> or sensitive characteristics. See
> [Responsible use](#responsible-use) below.

---

## Contents

- [Quick start](#quick-start)
- [What it does](#what-it-does)
- [Screenshots of the workflow](#the-five-views)
- [Architecture and data flow](#architecture-and-data-flow)
- [Supported formats](#supported-formats)
- [The synthetic corpus](#the-synthetic-corpus)
- [Configuration](#configuration)
- [Model choices, cost and privacy](#model-choices-cost-and-privacy)
- [Running the tests](#running-the-tests)
- [Evaluation](#evaluation)
- [Project layout](#project-layout)
- [RAG limitations](#rag-limitations)
- [Prompt injection and citation semantics](#prompt-injection-and-citation-semantics)
- [Responsible use](#responsible-use)
- [Troubleshooting](#troubleshooting)

---

## Quick start

Requires **Python 3.11 or newer**. No API key, no account and no cost.

```bash
git clone <your-repo-url> AI-Powered-Hiring-Assistant
cd AI-Powered-Hiring-Assistant
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. On first launch it generates the
synthetic corpus, embeds it and builds the index — about 30 seconds, mostly the
one-time download of the embedding model (~90 MB). Every later launch is
immediate.

Prefer `make`? The same thing:

```bash
make install
make run
```

**Verify the installation end to end:**

```bash
.venv/bin/python scripts/smoke_test.py
```

This drives the real pipeline and prints a pass/fail line for each capability.

### Platform notes

| Platform | Notes |
| --- | --- |
| **macOS** | Tested on Apple Silicon. Python 3.12 from Homebrew or python.org both work. `make venv` tries `python3.12` then falls back to `python3`. |
| **Linux** | Works as written. On a headless box, add `--server.headless=true`. |
| **Windows** | Use `.venv\Scripts\activate` instead of `source .venv/bin/activate`. If `faiss-cpu` fails to install, `pip install faiss-cpu --only-binary :all:` usually resolves it. The app runs without FAISS using a NumPy fallback. |

---

## What it does

| Capability | Detail |
| --- | --- |
| **Ingestion** | PDF, DOCX, TXT, Markdown, CSV and JSON. Validates type and size, extracts text with page/section provenance, records parsing warnings, and previews the extracted text. |
| **Chunking** | Structure-aware: splits on paragraphs, then sentences, packs to a configurable size with overlap, and never emits an empty fragment. |
| **Embeddings** | Configurable local `sentence-transformers` (default) or OpenAI. A deterministic hashing fallback keeps the app working with neither. |
| **Vector search** | FAISS with a persisted index, a stable chunk-id mapping and atomic rebuilds. Degrades to NumPy search if FAISS is unavailable. |
| **Semantic search** | Hybrid ranking (embedding similarity + term overlap), collection/type/document filters, near-duplicate collapsing, relevance display. |
| **Cited Q&A** | Answers constrained to retrieved passages. Every citation is validated; unresolvable ones are removed. If none survive, the answer is withheld. |
| **Abstention** | Detects when a question's vocabulary does not appear in the corpus and declines with an explanation, rather than answering from general knowledge. |
| **Profile summaries** | Cited summaries of a candidate's own documents. Verbatim extraction by default; optional model-drafted prose. |
| **Requirement matrix** | Maps each explicit job requirement to `Supported` / `Partial` / `Not found in supplied documents`, with evidence and citations. No aggregate score. |
| **Corpus management** | Inventory, indexing status, warnings, injection flags, per-document re-index and removal, index health, full rebuild. |
| **Session memory** | Per-session conversation history used only to resolve follow-up wording, with a one-click clear. |
| **Evaluation** | 25 deterministic cases across seven categories, with hit rate, recall, citation validity, abstention accuracy, injection resistance and latency. |
| **Export** | Cited answers, evidence matrices and evaluation reports to Markdown and CSV. |

---

## The five views

1. **Ask Assistant** — chat with collection filters, cited answers, expandable
   source excerpts, explicit low-evidence responses and a clear-chat action.
2. **Semantic Search** — query, top-k, filters, ranked snippets with source
   metadata, and a relevance indicator that is explicitly labelled a ranking
   signal rather than a probability.
3. **Candidate Explorer** — synthetic profiles, cited summaries and the job
   requirement evidence matrix. Never shows a suitability score or a verdict.
4. **Knowledge Base** — uploads, inventory, parse and index status, warnings,
   injection flags, remove/re-index and index health.
5. **Evaluation & Settings** — benchmark metrics with charts, provider and model
   status, retrieval configuration, and the safety documentation. Never exposes
   secrets.

---

## Architecture and data flow

```
UI (Streamlit)
   │
   ▼
HiringAssistant  ── the single seam between UI and pipeline
   │
   ├── Ingestion    validate → parse → scan → chunk → embed → index
   ├── Retrieval    embed query → FAISS → filter → rank → dedupe → coverage gate
   ├── Generation   prompt → extractive or LLM → validate citations → abstain if ungrounded
   ├── Evidence     requirement extraction → per-requirement evidence
   └── Storage      SQLite (metadata) + FAISS (vectors)
```

### Ingestion

```
file → validate → hash → parse → injection scan → chunk → embed → index → mark indexed
                     │
                     └── unchanged (same content, model and chunk config)? reuse and stop
```

The content hash covers the embedding model and chunking configuration as well
as the file bytes, so changing either correctly forces re-embedding instead of
reusing incomparable vectors.

### Query

```
question → guardrail screen → embed → FAISS (over-fetch) → metadata filter
        → hybrid rank → dedupe → relevance gate → corpus-gap gate
        → generate → validate citations → render
```

A failure at either gate produces an explicit abstention, not a weaker answer.

Full detail, including the rationale for each design decision, is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Supported formats

| Format | Extension | Extraction |
| --- | --- | --- |
| PDF | `.pdf` | `pypdf`, page by page. Scanned PDFs produce a warning — OCR is out of scope. |
| Word | `.docx` | `python-docx`. Headings become sections; tables are extracted as pipe-delimited text. |
| Plain text | `.txt` | Block-split on blank lines. |
| Markdown | `.md`, `.markdown` | Split on headings, so each section is citable. |
| CSV | `.csv` | Each row rendered as `Header: value` blocks, so a row is a coherent retrievable passage. |
| JSON | `.json` | Nested structures flattened to indented key/value text, one passage per record. |

Size limit: 25 MB per file (`MAX_UPLOAD_MB`). Empty files, binary content with
a text extension, malformed JSON, truncated PDFs and invalid DOCX packages are
rejected with a specific message and a remedy.

---

## The synthetic corpus

The bundled demo data is **entirely fictional and machine-generated from a fixed
seed** (`20261011`), so it is byte-identical on every machine.

| Item | Count |
| --- | --- |
| Candidate profiles | 24, across 15 role archetypes |
| Job descriptions | 7 |
| Hiring / process / policy documents | 8 |
| Formats exercised | CSV, JSON, Markdown, TXT, PDF, DOCX |
| Malformed test fixtures | 8 |

The corpus is designed so that retrieval and evaluation are *checkable*:

- Every profile has a stable, greppable id, and skills come from a fixed
  vocabulary — so a paraphrase query ("data visualisation") has a known correct
  source ("Data Visualization").
- Three profiles are deliberately adjacent (overlapping skills), so ranking is
  genuinely non-trivial.
- Two documents deliberately **conflict** on an interview timeline, so the
  "show both sources rather than silently picking one" behaviour is exercised by
  real corpus content.
- One hiring document contains an **embedded prompt-injection attempt**, so the
  adversarial path is tested against genuine corpus content, not just a
  synthetic test string.

**No real candidate data is used, and the corpus contains no protected
attributes at all** — a test asserts no candidate record carries a field for
age, gender, marital status, religion, caste, nationality, disability, health
and the like.

Regenerate it at any time:

```bash
make corpus
# or
.venv/bin/python scripts/build_corpus.py --force --candidates 30 --seed 12345
```

---

## Configuration

Copy `.env.example` to `.env`. Every value has a working default, so an empty
`.env` runs the full application locally.

```bash
cp .env.example .env
```

The most useful settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `local-minilm` | `local-minilm`, `local-mpnet`, `openai-small`, `openai-large` |
| `GENERATION_MODEL` | `extractive` | `extractive`, `openai-mini`, `openai-4o` |
| `OPENAI_API_KEY` | *(empty)* | Required only for OpenAI models |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | Changing either invalidates the index |
| `TOP_K` | `6` | Passages retrieved per query |
| `MIN_RELEVANCE` | `0.18` | Below this the assistant abstains |
| `MAX_UPLOAD_MB` | `25` | Per-file upload limit |

Configuration is read from the environment, not the UI, so a running session
cannot silently change how documents were embedded. Change `.env` and restart.

### Rebuilding the index

Needed after changing the embedding model, chunk size or overlap:

```bash
make index
# or
.venv/bin/python scripts/build_index.py --rebuild
```

The rebuild constructs the new index, validates it by reading it back, and only
then swaps it into place — a failure leaves the previous working index intact.

---

## Model choices, cost and privacy

### Local (default) — free, offline, private

`sentence-transformers/all-MiniLM-L6-v2` (384-d) for embeddings, extractive
generation for answers.

- No credentials, no cost, no network calls after the one-time model download.
- **No document or query text leaves the machine.**
- Extractive mode quotes retrieved passages verbatim rather than paraphrasing
  them. It is honest about what it is and every line is directly checkable
  against its source.

### OpenAI (opt-in)

Set `OPENAI_API_KEY` and select an OpenAI model. Then:

| Sent | When |
| --- | --- |
| Every document chunk | During ingestion, for embeddings |
| Every question | Per query, for the query embedding |
| Question + retrieved passages | Per query, for generation |

**Cost** is per token on both ingestion and queries. **Privacy:** document text
is processed by a third party, which for candidate documents is third-party
processing under most data-protection regimes and may require a DPA and
disclosure. The app discloses this in the header, the sidebar and the upload
panel, and never enables it implicitly.

Requesting an OpenAI model without a key falls back to local operation and
records the fallback visibly in the UI.

Full detail: [docs/PRIVACY_AND_RESPONSIBLE_USE.md](docs/PRIVACY_AND_RESPONSIBLE_USE.md).

---

## Running the tests

```bash
make test          # or: .venv/bin/python -m pytest tests/ -q
make test-cov      # with coverage
make check         # lint + tests
```

The suite runs **offline, with no model download and no API key**, using a
deterministic hashing embedder that exercises the same code paths as the real
model.

| File | Covers |
| --- | --- |
| `tests/test_ingestion.py` | Validation, all parsers, chunking, metadata, dedupe, removal |
| `tests/test_retrieval.py` | Embeddings, FAISS save/load, mismatch detection, filtering, ranking |
| `tests/test_generation.py` | Citation validation, abstention, guardrails, provider failures, memory |
| `tests/test_evidence.py` | Requirement extraction, matrix wording, summaries, injection, benchmark |
| `tests/test_corpus_and_config.py` | Corpus determinism, protected-attribute absence, config, storage |

Current status: **181 tests passing**, `ruff` clean.

---

## Evaluation

```bash
make eval
```

Runs 25 deterministic cases and writes
[docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md) and a matching CSV.
The latest recorded run:

| Metric | Value |
| --- | --- |
| Pass rate | **100%** (25/25) |
| Hit rate / Recall@k | **100%** |
| Citation validity | **100%** |
| Abstention accuracy | **100%** |
| Injection resistance | **100%** |
| Precision@k | 58.3% |
| Latency p50 / p95 | 16 ms / 20 ms |
| Error rate | 0% |

Precision@k is deliberately not 100% and that is correct — a question like
"what does this role require?" legitimately retrieves several passages from one
document. Recall is the metric that matters for retrieval: an unretrieved
passage cannot be cited. See
[docs/EVALUATION.md](docs/EVALUATION.md) for the full breakdown, the case
catalogue, and an honest discussion of what these numbers do and do not show.

---

## Project layout

```
AI-Powered-Hiring-Assistant/
├── app.py                     Streamlit entry point
├── src/hiring_assistant/
│   ├── config.py              All configuration, with local fallbacks
│   ├── errors.py              Structured errors carrying a remedy
│   ├── models.py              Domain dataclasses
│   ├── services.py            The UI ⇄ pipeline seam
│   ├── synthetic.py           Deterministic corpus generator
│   ├── _compat.py             Native library import ordering
│   ├── ingestion/             parsers · chunking · pipeline
│   ├── retrieval/             embeddings · search
│   ├── generation/            prompts · citations · generator
│   ├── evidence/              requirements · profiles
│   ├── safety/                guardrails · injection
│   ├── storage/               database · vector_store
│   ├── evaluation/            cases · runner · reporting
│   └── ui/                    theme · components · sidebar · views/
├── data/                      index, database, synthetic corpus (regenerable)
├── docs/                      ARCHITECTURE · EVALUATION · PRIVACY · DATA_DICTIONARY
├── scripts/                   build_corpus · build_index · run_evaluation · smoke_test · make_zip
├── tests/                     181 tests across five files
├── Makefile · pyproject.toml · requirements.txt · .env.example · .gitignore
```

`data/index`, `data/*.sqlite3` and `data/synthetic` are all regenerable and are
excluded from version control and from the distribution ZIP.

---

## RAG limitations

Worth knowing before trusting any output:

- **"Not found" is not "does not exist."** It means the supplied documents
  contain no evidence. A candidate may have the skill and simply not have
  mentioned it. The UI states this on every page that shows a matrix.
- **Retrieval is lexical-semantic, not reasoning.** If the answer requires
  combining three documents in a way no single passage expresses, retrieval may
  not surface the right evidence.
- **Scanned PDFs are not supported.** They need OCR, which this MVP does not
  perform. The parser warns when a majority of pages yield no text.
- **Chunk boundaries can split an answer.** Overlap mitigates this but does not
  eliminate it.
- **Similarity is not truth.** The relevance figure is a ranking signal. It is
  not calibrated and is never presented as confidence.
- **The corpus is synthetic.** The benchmark demonstrates that the pipeline
  behaves as specified. It is not a claim about performance on real recruitment
  data.
- **Conflicting sources are surfaced, not resolved.** When two documents
  disagree the assistant cites both and says they disagree. It does not decide
  which is right.

---

## Prompt injection and citation semantics

### Documents are untrusted data

A resume or vendor questionnaire may contain text designed to hijack the
assistant. The app detects five pattern families — instruction override, prompt
exfiltration, tool escalation, role hijack and delimiter injection — and on a
match:

1. **Nothing is executed.** There is no code path from document content to a
   shell command, a network request or a tool call.
2. The flagged span is replaced with an inert marker before reaching the model.
3. The document is flagged visibly in the Knowledge Base.
4. The system policy tells the model that a command inside the data envelope is
   suspicious content to report, not an instruction to follow.

The bundled corpus contains a real injection attempt so this is tested against
genuine content.

### Citations mean something specific

- A citation renders only if its marker resolves to a passage retrieved **for
  that exact query**.
- A quoted span is shown only if it can be located in the cited passage.
  Fabricated quotes are dropped and the passage's own excerpt is shown instead.
- If no citation survives validation, the answer is **withheld** and the
  assistant abstains.

This is why citation validity is structurally 100%: an unresolvable citation
cannot be rendered.

---

## Responsible use

**This system retrieves and summarises information. It does not make employment
decisions.**

It will not:

- produce a hire/reject recommendation or an accept/reject decision
- produce a suitability score, rating or grade for a candidate
- rank or order candidates
- assess age, gender, marital or family status, religion, caste, nationality,
  disability, health, or any other protected characteristic
- assess personality, culture fit or "potential"

These are enforced in code and asserted by tests, not merely documented. The
requirement matrix has no aggregate field at all — a test asserts the serialised
payload exposes no `score`, `total`, `rank` or `verdict` key.

Before using real candidate data, read
[docs/PRIVACY_AND_RESPONSIBLE_USE.md](docs/PRIVACY_AND_RESPONSIBLE_USE.md). It
covers lawful basis, minimisation, retention, access control, the provider
decision and the Phase 2 roadmap for each.

---

## Troubleshooting

**The app starts but shows "No index yet."**
Click **Load synthetic corpus** in the sidebar or the Knowledge Base tab. On a
genuinely fresh install this happens automatically; if it did not, check the
terminal for an error.

**"Index / model mismatch."**
The saved index was built with a different embedding model or chunk size. This
is intentional — vectors from different models are not comparable. Rebuild:
`make index`.

**The first launch is slow.**
It is downloading the embedding model (~90 MB) from Hugging Face. Subsequent
launches are immediate. To avoid the download entirely, install without
`sentence-transformers`; the app falls back to a deterministic lexical embedder
and says so in the UI.

**Search results look poor.**
Check the Settings tab: if the model reads "Hashing fallback", the real
embedding model is unavailable. Install `sentence-transformers` and restart.

**`ImportError: No module named 'hiring_assistant'`.**
Run from the project root, or use `make run`. `app.py` adds `src/` to the path
itself; scripts do the same.

**Streamlit shows a blank page or a stale UI.**
Stop the server (Ctrl-C) and restart. Streamlit caches imports between reruns.

**A PDF produces no text.**
It is almost certainly a scan. The parser warns explicitly. OCR is out of scope
for this MVP — supply a text-based PDF or another format.

**`faiss` fails to install (often on Windows).**
`pip install faiss-cpu --only-binary :all:`. The app also runs without FAISS,
using a NumPy inner-product search with identical semantics; the active backend
is shown in the Settings tab.

**Everything is slow on an external provider.**
Generation latency dominates and is network-bound. Switch
`GENERATION_MODEL=extractive` for local, instant answers.

**I want a completely clean slate.**
```bash
make clean-all
```
Removes the index, database and generated corpus. The next launch regenerates
all three.

---

## Licence and provenance

Portfolio project built to the specification in
`AI_Powered_Hiring_Assistant_PRD.docx`. All bundled data is synthetic and
machine-generated. Built with Streamlit, FAISS, sentence-transformers, pypdf,
python-docx, Plotly and pytest.
