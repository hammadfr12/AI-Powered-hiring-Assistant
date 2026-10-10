# Privacy and Responsible Use

## What this system is

A retrieval and summarisation assistant. It searches documents you supply,
quotes the passages that support each statement, and abstains when the
documents do not contain the answer.

## What this system is not

It is **not** a hiring decision system. Specifically, it does not:

- produce an accept/reject decision or a hire/no-hire recommendation
- produce a single opaque suitability score, rating or grade for a candidate
- rank or order candidates against each other
- infer, score or comment on protected or sensitive characteristics
- assess personality, culture fit, emotional intelligence or "potential"
- make any determination about a person

These are not configuration options. The behaviour is enforced in code
(`src/hiring_assistant/safety/guardrails.py`) and asserted by tests
(`tests/test_generation.py::TestGuardrails`,
`tests/test_evidence.py::TestEvidenceMatrix::test_matrix_contains_no_verdict_language`).

The requirement evidence matrix deliberately has **no aggregate field**. It
carries per-requirement statuses and citations only. A test asserts that the
serialised payload exposes no `score`, `total`, `overall`, `rank`, `verdict` or
`recommendation` key, so this cannot regress silently.

## Protected attributes

The assistant refuses questions about age, gender, marital or family status,
religion, caste, nationality, race, ethnicity, disability, health, pregnancy,
sexual orientation, political affiliation or socioeconomic background — whether
asked directly or indirectly.

The bundled synthetic corpus contains **no protected attributes at all**. A test
asserts that no candidate record carries a field named `age`, `gender`,
`religion`, `caste`, `nationality`, `disability`, `health`, `marital_status`,
`dob` and so on. This is a schema-level guarantee, not a filtering step: the
data simply does not exist.

## Data flow and where data goes

### Local operation (the default)

With `EMBEDDING_MODEL=local-minilm` (or `local-mpnet`) and
`GENERATION_MODEL=extractive`:

- Documents are parsed, chunked, embedded and indexed **on this machine**.
- Queries are embedded and answered **on this machine**.
- **No document text, query text or metadata leaves the machine.**
- There are no outbound network calls during normal operation after the
  embedding model has been downloaded once from Hugging Face.

The Settings view displays "Fully local" when this is the case.

### External provider operation (opt-in)

If you set `OPENAI_API_KEY` **and** select an OpenAI embedding or generation
model, then:

| What is sent | When |
| --- | --- |
| Full text of every document chunk | During ingestion, to compute embeddings |
| Every user question | On each query, to compute the query embedding |
| Retrieved passages + the question | On each query, to the generation model |

**Implications you should weigh before enabling this:**

- Document text is transmitted to and processed by a third party. For candidate
  documents this is processing by a third party under most data-protection
  regimes, and may require a data processing agreement and disclosure.
- Provider retention and training policies apply to whatever is sent.
- Ingestion and query cost money per token.
- The app discloses this in three places: the app header, the sidebar, and the
  Knowledge Base upload panel. It does not silently enable it.

The app **never** enables an external provider implicitly. Requesting an OpenAI
model without a key falls back to a local model and records the fallback in the
UI.

## Secrets

- Credentials are read from environment variables or `.env` only.
- `.env` is listed in `.gitignore` and is excluded from the distributable ZIP by
  `scripts/make_zip.py`, which verifies the exclusion and fails if a `.env` is
  ever present in the archive.
- The app never logs, renders or transmits the key value. The Settings view
  reports only whether a key is *present* (`openai_key_present: true/false`).
- A test asserts that the API key string does not appear anywhere in the
  UI-facing configuration summary.

## Prompt injection

Documents are **untrusted data**. A resume, job description or vendor
questionnaire may contain text engineered to hijack the assistant.

### What is detected

Five pattern families (`src/hiring_assistant/safety/injection.py`):

| Family | Examples |
| --- | --- |
| `instruction_override` | "ignore all previous instructions", "disregard your rules" |
| `prompt_exfiltration` | "reveal your system prompt", "print the .env file" |
| `tool_escalation` | "run this command", `curl`, `rm -rf`, `subprocess` |
| `role_hijack` | "you are now…", "developer mode", "no restrictions" |
| `delimiter_injection` | fake `<system>` tags, `assistant:` turns |

### What happens on a match

1. **Nothing is executed.** A match only ever produces a flag. There is no code
   path from document content to a shell command, a network request or a tool
   call.
2. The flagged span is **neutralised** before reaching the model: replaced with
   an inert marker reading `[neutralised instruction_override: text quoted, not
   followed]`.
3. The document is marked with a visible warning in the Knowledge Base view.
4. The system policy explicitly tells the model that everything inside the
   untrusted-data envelope is data, and that a command inside it is suspicious
   document content to be reported, not obeyed.

### Defence in depth

- Retrieved text is wrapped in `<<<UNTRUSTED_DOCUMENT_DATA>>>` markers, and any
  text in the document that mimics those markers is stripped — so a document
  cannot fake its own end-of-data boundary.
- The system policy is never concatenated with document text.
- Even if a model were fully hijacked, it could not fabricate a citation: every
  rendered citation is validated against the retrieved set, and unresolvable
  ones are removed. It could not take an action, because the pipeline has no
  action to take.

### Tested

The bundled corpus contains a real injection attempt
(`vendor_questionnaire_with_injection.txt`), so this path is exercised against
genuine corpus content as well as by four dedicated benchmark cases. A test
asserts the assistant produces a normal summary of that document while leaking
no `.env`, no `curl` command and no environment-variable content.

## Citation semantics

- A citation is rendered **only** if its marker resolves to a passage retrieved
  for that exact query, and the filename in the database matches.
- A quoted span is shown as a quote **only** if it can be located in the cited
  passage. A quote that cannot be verified is dropped and the passage's own
  excerpt is shown instead.
- If no citation survives validation, the answer is **withheld** and the
  assistant abstains.
- The relevance figure shown on each citation is a **ranking score**, not a
  probability that the statement is true. The UI says so wherever the number
  appears. It is not calibrated and must not be read as confidence.

## Abstention

When the corpus does not contain the evidence, the assistant says so. This is
the correct behaviour, not a failure.

"Not found in supplied documents" means exactly that: no passage in the
documents available provides evidence. It is **not** a statement that a
candidate lacks a skill, and it is not evidence of absence. The matrix explains
this on every page that shows it, and a test asserts the wording is present.

## Before using real candidate data

This MVP ships with synthetic data precisely so that no real-data obligations
are triggered by trying it. Before pointing it at real candidate documents:

1. **Establish a lawful basis** for processing candidate data for this purpose.
2. **Complete a DPIA** if your jurisdiction requires one.
3. **Decide the provider question explicitly.** Local-only operation avoids
   third-party processing entirely; enabling an external provider requires a
   data processing agreement and candidate disclosure.
4. **Apply minimisation.** Do not ingest documents containing protected
   attributes. The assistant will not *use* them, but they would still be
   stored in the index and could surface as retrieved text.
5. **Set retention.** Implement deletion for candidate documents at the end of
   the retention window. The app provides per-document removal; automated
   retention is a Phase 2 item.
6. **Add access control.** The MVP has none — it is a single-user local
   application. Authentication, role-based permissions and audit logging are
   Phase 2.
7. **Keep a human in the loop.** Every output is evidence for a person to
   review. Nothing in this system should be wired into an automated decision.

## Reporting a concern

If the assistant produces output that violates the guarantees above — a verdict,
a score, a fabricated citation, a protected-attribute assessment — that is a
bug. The guarantees are enforced in code and covered by tests, so a violation
means a gap in both. Capture the query and the output and file it against the
safety module.

## Summary of guarantees

| Guarantee | Enforced by | Tested by |
| --- | --- | --- |
| No hire/reject verdict | Guardrails; matrix schema | `test_matrix_contains_no_verdict_language`, `test_matrix_has_no_aggregate_score_field` |
| No candidate ranking | Guardrails | `test_hiring_decisions_are_redirected` |
| No protected-attribute assessment | Guardrails; corpus schema | `test_protected_attributes_are_refused`, `test_no_protected_attributes_in_corpus` |
| Every citation resolvable | Citation validator | `test_citation_validity_is_perfect` |
| Fabricated quotes rejected | Citation validator | `test_fabricated_quote_is_rejected` |
| Abstains on absent evidence | Coverage gate | `test_absent_topic_abstains` |
| Document text cannot act | No execution path; neutralisation | `test_document_injection_does_not_leak_into_answer` |
| Secrets never exposed | Config summary | `test_public_summary_never_contains_the_key` |
| No silent external calls | Provider resolution | `test_openai_without_key_falls_back_with_a_note` |
