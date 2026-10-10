# AI-Powered Hiring Assistant

A recruiter-facing retrieval-augmented generation (RAG) application for natural-language search and evidence-grounded Q&A across candidate profiles, job descriptions and approved hiring documents.

Every statement it makes carries a citation that resolves to a real retrieved passage. When the documents do not contain the answer, it says so instead of guessing.

> **This is an information retrieval assistant, not a hiring decision system.** It does not score, rank or select candidates, and it will not assess protected or sensitive characteristics. See [Responsible use](#responsible-use) below.

---

## Quick start

Requires Python 3.11 or newer. No API key, account or cost is required for the default local configuration.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. On first launch it generates a synthetic corpus and builds the search index. Configuration options are documented in `.env.example`.

## Features

- Ingestion of PDF, DOCX, TXT, Markdown, CSV and JSON documents.
- Structure-aware chunking with source provenance.
- Local embeddings by default, with optional OpenAI-compatible providers.
- Hybrid semantic retrieval, filters, and near-duplicate collapsing.
- Evidence-grounded question answering with validated citations and abstention when sources do not support an answer.
- Candidate-document summaries and a requirement-to-evidence matrix without aggregate candidate scoring.
- Corpus management, injection warnings, indexing health and evaluation tools.

## Responsible use

This tool supports document retrieval and evidence inspection. It is not designed to make, automate or recommend hiring decisions. Do not use it to infer protected or sensitive traits or as a substitute for accountable human review. Use only documents you are authorised to process, and review `docs/PRIVACY_AND_RESPONSIBLE_USE.md` before handling real data.

## Documentation

- `docs/ARCHITECTURE.md` — architecture and data flow
- `docs/DATA_DICTIONARY.md` — stored entities and fields
- `docs/EVALUATION.md` and `docs/EVALUATION_RESULTS.md` — evaluation approach and results
- `docs/PRIVACY_AND_RESPONSIBLE_USE.md` — privacy and responsible-use guidance

## Tests and evaluation

Install dependencies, then run:

```bash
python -m pytest
python scripts/smoke_test.py
python scripts/run_evaluation.py
```

See the project `Makefile` for additional tasks. Generated indexes, local databases, virtual environments and secrets should remain untracked; see `.gitignore`.
