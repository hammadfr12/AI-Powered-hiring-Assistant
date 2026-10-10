# Data Dictionary

Two stores hold application state:

- **SQLite** (`data/hiring_assistant.sqlite3`) — documents, chunks, conversations,
  messages, evaluation cases and runs, key/value state.
- **FAISS index** (`data/index/`) — vectors plus a JSON sidecar describing how
  the index was built.

Everything in both stores is **regenerable**. Deleting them and reloading the
corpus rebuilds identical state, because the corpus is generated from a fixed
seed.

---

## SQLite schema

Schema version 1. Created by `MetadataStore._initialise`.

### `documents`

One row per source file that has been submitted for ingestion.

| Column | Type | Notes |
| --- | --- | --- |
| `document_id` | TEXT PK | `doc_<12 hex>`, generated on ingest |
| `filename` | TEXT | Sanitised display name; may include a relative path for corpus files |
| `file_type` | TEXT | Extension without the dot: `pdf`, `docx`, `txt`, `md`, `csv`, `json` |
| `collection` | TEXT | `candidate_profiles`, `job_descriptions`, `hiring_documents`, `uploads`, `general` |
| `source_hash` | TEXT | SHA-256 over **content + embedding model + chunk size + overlap** |
| `char_count` | INTEGER | Characters extracted after normalisation |
| `chunk_count` | INTEGER | Passages produced |
| `page_count` | INTEGER NULL | PDF page count, or DOCX section count; `NULL` for text formats |
| `status` | TEXT | `pending` · `indexed` · `skipped` · `failed` · `removed` |
| `warnings` | TEXT (JSON) | Array of parsing/injection notes shown in the UI |
| `error` | TEXT NULL | Failure message when `status = 'failed'` |
| `synthetic` | INTEGER | 1 for generated demo data, 0 for user uploads |
| `origin` | TEXT | `synthetic` or `upload` |
| `created_at` | TEXT | ISO-8601 UTC, `Z` suffix |
| `updated_at` | TEXT | ISO-8601 UTC |
| `indexed_at` | TEXT NULL | Set when the document first reached `indexed` |

Indexes: `collection`, `source_hash`, `status`.

> **`source_hash` deliberately includes the embedding model and chunking
> configuration.** Hashing content alone would cause the pipeline to reuse
> vectors that were built under a different model — incomparable with a query
> embedded by the current one. Including them makes "unchanged" mean
> "unchanged *and* still comparable".

### `chunks`

One row per retrievable passage. The unit of citation.

| Column | Type | Notes |
| --- | --- | --- |
| `chunk_id` | TEXT PK | `{document_id}::c{index:04d}` — deterministic |
| `document_id` | TEXT FK | → `documents.document_id`, `ON DELETE CASCADE` |
| `chunk_index` | INTEGER | 0-based position within the document, contiguous |
| `text` | TEXT | The passage, whitespace-normalised, non-empty |
| `filename` | TEXT | Denormalised from the document for display without a join |
| `collection` | TEXT | Denormalised; used as a retrieval pre-filter |
| `file_type` | TEXT | Denormalised |
| `section` | TEXT NULL | Heading or `Page N` / `Row N` / `Table N` |
| `page` | INTEGER NULL | PDF page number when known |
| `char_start` | INTEGER | Offset into the reconstructed document text |
| `char_end` | INTEGER | End offset |
| `embedding_model` | TEXT | Model id that produced the vector |
| `index_id` | INTEGER NULL | Row position in the FAISS index |
| `synthetic` | INTEGER | 1 for demo data |
| `token_estimate` | INTEGER | `len(text) // 4` — approximate, used for context budgeting |

Indexes: `document_id`, `index_id`, `collection`.

> **`chunk_id` is deterministic** and `index_id` is the FAISS row position.
> They are kept in step by `MetadataStore.set_index_ids`. A chunk id that
> resolves in SQLite but has no `index_id` is a consistency error the health
> check surfaces.

### `conversations`

Session scope for chat state.

| Column | Type | Notes |
| --- | --- | --- |
| `session_id` | TEXT PK | Streamlit session key |
| `title` | TEXT | Auto-set from the first question |
| `created_at` | TEXT | ISO-8601 UTC |
| `updated_at` | TEXT | Bumped on every message |

### `messages`

| Column | Type | Notes |
| --- | --- | --- |
| `message_id` | TEXT PK | `msg_<12 hex>` |
| `session_id` | TEXT FK | → `conversations.session_id`, cascade delete |
| `role` | TEXT | `user` · `assistant` · `system` |
| `content` | TEXT | Full message text |
| `cited_chunk_ids` | TEXT (JSON) | Array of chunk ids cited by this message |
| `metadata` | TEXT (JSON) | `abstained`, `blocked`, `mode`, `latency_ms` |
| `created_at` | TEXT | ISO-8601 UTC |

Index: `(session_id, created_at)`.

> Memory is scoped to a session and is passed to the model only as a
> follow-up-resolution hint, explicitly labelled as **not evidence**. Clearing
> deletes the rows; there is no soft-delete.

### `evaluation_cases`

The benchmark definition, persisted so a run can be reproduced from the store.

| Column | Type | Notes |
| --- | --- | --- |
| `case_id` | TEXT PK | e.g. `ev_direct_01` |
| `query` | TEXT | The question asked |
| `category` | TEXT | One of the seven categories |
| `expected_source_ids` | TEXT (JSON) | Filenames or filename prefixes |
| `expected_answer_points` | TEXT (JSON) | Recorded for human review; not auto-scored |
| `should_abstain` | INTEGER | Expected abstention |
| `expected_blocked` | INTEGER | Expected guardrail refusal |
| `notes` | TEXT | Why the case exists |
| `candidate_id` | TEXT NULL | Reserved for candidate-scoped cases |
| `job_id` | TEXT NULL | Reserved for job-scoped cases |

### `evaluation_runs`

| Column | Type | Notes |
| --- | --- | --- |
| `run_id` | TEXT PK | `eval_<12 hex>` |
| `created_at` | TEXT | ISO-8601 UTC |
| `config` | TEXT (JSON) | top_k, min_relevance, models, chunking, case count |
| `metrics` | TEXT (JSON) | The aggregate metric dictionary |
| `failures` | TEXT (JSON) | `"<case_id> (<category>): <reason>"` strings |
| `duration_seconds` | REAL | Wall-clock duration |
| `results` | TEXT (JSON) | Full per-case result array |

Index: `created_at`.

### `index_state`

Key/value store for derived state.

| Column | Type | Notes |
| --- | --- | --- |
| `key` | TEXT PK | e.g. `index_built_at`, `index_vector_count` |
| `value` | TEXT (JSON) | JSON-encoded value |
| `updated_at` | TEXT | ISO-8601 UTC |

### `meta`

| Column | Type | Notes |
| --- | --- | --- |
| `key` | TEXT PK | Currently only `schema_version` |
| `value` | TEXT | `"1"` |

---

## FAISS index directory

`data/index/`

| File | Purpose |
| --- | --- |
| `faiss.index` | The FAISS `IndexFlatIP` binary (inner product over L2-normalised vectors, so IP ≡ cosine) |
| `vectors.npy` | Fallback store when FAISS is unavailable — same semantics, NumPy search |
| `index_meta.json` | Sidecar: build configuration **and** the chunk-id order |
| `backend.txt` | `faiss` or `numpy` |

### `index_meta.json` structure

```json
{
  "metadata": {
    "embedding_model": "local-minilm",
    "embedding_model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 384,
    "chunk_size": 900,
    "chunk_overlap": 150,
    "backend": "faiss",
    "vector_count": 309,
    "built_at": "2026-10-11T02:20:00Z",
    "corpus_fingerprint": "sha256…",
    "schema_version": 1
  },
  "chunk_ids": ["doc_abc123::c0000", "doc_abc123::c0001", "…"]
}
```

**Row *i* of the index corresponds to `chunk_ids[i]`.** This is the stable
mapping between a vector and the passage it represents, and it is why the
sidecar must be written and read atomically with the vectors.

### Mismatch detection

On load, `IndexMetadata.matches()` compares the sidecar against
`Settings.index_fingerprint()` on: `embedding_model`, `embedding_model_id`,
`dimension`, `chunk_size`, `chunk_overlap`.

A mismatch raises `IndexMismatchError` with a message naming the specific field
that differs, rather than silently returning meaningless results. Vectors from
two different models are not comparable, and a chunk-size change alters what
each row *means*.

---

## Synthetic corpus

`data/synthetic/` (regenerable; excluded from the ZIP)

| Path | Contents |
| --- | --- |
| `candidates.csv` | 24 candidates, compact tabular index |
| `candidates.json` | Same fields as CSV, plus a pointer to the full profile file |
| `job_descriptions.json` | All 7 job descriptions as structured records |
| `profiles/cand_NNN_<name>.txt` | One readable profile document per candidate |
| `jobs/job_<slug>.md` | One readable job description per role |
| `documents/*.{md,txt,pdf,docx}` | 8 hiring/process/policy documents across four formats |
| `corpus_manifest.json` | Version, seed, counts and id lists |
| `_malformed/*` | Deliberately broken fixtures for tests. **Never loaded** — the loader skips this directory |

### Candidate record fields

| Field | Notes |
| --- | --- |
| `profile_id` | `cand_NNN` |
| `full_name` | Fictional name |
| `role_title` | One of 15 role archetypes |
| `seniority` | `Associate` · `Mid-level` · `Senior` · `Lead` |
| `location` | Fictional city |
| `education` | Degree label |
| `years_experience` | Derived from the seniority band |
| `experience_start_year`, `experience_end_year` | Explicit dates, so any experience figure is transparently derived |
| `core_skills`, `additional_skills` | From a fixed 56-skill vocabulary |
| `projects` | Three narrative project statements |
| `certifications` | Fictional certificate name or `None listed` |
| `availability_note` | Working-preference statement |
| `summary` | One-paragraph characterisation |
| `synthetic` | Always `true` |

**Deliberately absent:** age, date of birth, gender, marital or family status,
religion, caste, nationality, race, ethnicity, disability, health, photograph,
political affiliation. A test asserts no candidate record carries any of these
fields.

### Corpus determinism

`generate_corpus(seed=20261011, candidate_count=24)` is a pure function. The
generator uses a hand-rolled LCG (`utils.DeterministicRandom`) rather than
`random.Random`, whose output is not guaranteed stable across Python major
versions. Two calls with the same arguments produce byte-identical files on any
machine.

`corpus_is_stale()` compares the on-disk manifest against the current
`CORPUS_VERSION`, seed and candidate count, so regenerating is automatic when
any of them changes.

---

## Entity relationships

```
documents 1 ──── * chunks
     │                 │
     │                 └── index_id ──► FAISS row position
     │
conversations 1 ──── * messages ── cited_chunk_ids ──► chunks.chunk_id

evaluation_cases ──(run)──► evaluation_runs ── results[].retrieved_source_ids ──► documents.filename

index_state ── derived bookkeeping (built_at, vector_count)
```

## Data retention and deletion

| Action | Effect |
| --- | --- |
| Remove one document (Knowledge Base) | Deletes the `documents` row, cascades to `chunks`, physically removes the vectors from FAISS and saves the index atomically |
| Clear conversation (Ask Assistant) | Deletes the `messages` rows for that session; the conversation row is kept with a reset title |
| Clear the entire knowledge base (Settings) | Deletes all documents, chunks and vectors; clears `index_state` |
| Delete `data/` entirely | Full reset; the next launch regenerates the corpus and index |

There is no soft delete and no tombstone. Removed data cannot be retrieved.
