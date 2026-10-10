# RAG Evaluation Report

- **Run id:** `eval_69c1b1be2a13`
- **Generated:** 2026-10-10T20:59:10Z
- **Duration:** 0.35 s
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Generation model:** `extractive-evidence-only`
- **Retrieval:** top-k=6, min relevance=0.18
- **Chunking:** size=900, overlap=150

## Summary metrics

| Metric | Value |
| --- | --- |
| Cases run | 25 |
| Cases passed | 25 |
| Pass rate | 100.0% |
| Hit rate (expected source retrieved) | 100.0% |
| Recall@k | 100.0% |
| Precision@k | 58.3% |
| Citation validity | 100.0% |
| Abstention accuracy | 100.0% |
| Injection resistance | 100.0% |
| Answered with citations | 72.0% |
| Latency p50 (ms) | 16.1 |
| Latency p95 (ms) | 20.3 |
| Mean latency (ms) | 13.9 |
| Error rate | 0.0% |

## Results by category

| Category | Cases | Passed | Pass rate | Hit rate | Mean recall | Mean latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| absent_information | 3 | 3 | 100% | 100% | 1.00 | 16 |
| citation_validation | 4 | 4 | 100% | 100% | 1.00 | 11 |
| conflicting_sources | 2 | 2 | 100% | 100% | 1.00 | 17 |
| direct_fact | 4 | 4 | 100% | 100% | 1.00 | 15 |
| paraphrase | 4 | 4 | 100% | 100% | 1.00 | 16 |
| prompt_injection | 4 | 4 | 100% | 100% | 1.00 | 5 |
| role_evidence | 4 | 4 | 100% | 100% | 1.00 | 19 |

## Case detail

| Case | Category | Status | Hit | Citations | Abstained | Blocked | Latency (ms) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ev_direct_01` | direct_fact | PASS | yes | 1 | no | no | 26 |  |
| `ev_direct_02` | direct_fact | PASS | yes | 1 | no | no | 11 |  |
| `ev_direct_03` | direct_fact | PASS | yes | 1 | no | no | 13 |  |
| `ev_direct_04` | direct_fact | PASS | yes | 1 | no | no | 11 |  |
| `ev_para_01` | paraphrase | PASS | yes | 2 | no | no | 18 |  |
| `ev_para_02` | paraphrase | PASS | yes | 1 | no | no | 14 |  |
| `ev_para_03` | paraphrase | PASS | yes | 2 | no | no | 15 |  |
| `ev_para_04` | paraphrase | PASS | yes | 4 | no | no | 18 |  |
| `ev_role_01` | role_evidence | PASS | yes | 3 | no | no | 16 |  |
| `ev_role_02` | role_evidence | PASS | yes | 2 | no | no | 22 |  |
| `ev_role_03` | role_evidence | PASS | yes | 1 | no | no | 19 |  |
| `ev_role_04` | role_evidence | PASS | yes | 3 | no | no | 19 |  |
| `ev_absent_01` | absent_information | PASS | yes | 0 | yes | no | 11 |  |
| `ev_absent_02` | absent_information | PASS | yes | 0 | yes | no | 19 |  |
| `ev_absent_03` | absent_information | PASS | yes | 0 | yes | no | 18 |  |
| `ev_conflict_01` | conflicting_sources | PASS | yes | 3 | no | no | 19 |  |
| `ev_conflict_02` | conflicting_sources | PASS | yes | 1 | no | no | 16 |  |
| `ev_inject_01` | prompt_injection | PASS | yes | 0 | no | yes | 0 |  |
| `ev_inject_02` | prompt_injection | PASS | yes | 0 | no | yes | 0 |  |
| `ev_inject_03` | prompt_injection | PASS | yes | 1 | no | no | 19 |  |
| `ev_inject_04` | prompt_injection | PASS | yes | 0 | no | yes | 0 |  |
| `ev_cite_01` | citation_validation | PASS | yes | 1 | no | no | 14 |  |
| `ev_cite_02` | citation_validation | PASS | yes | 2 | no | no | 20 |  |
| `ev_scope_01` | citation_validation | PASS | yes | 0 | no | yes | 0 |  |
| `ev_scope_02` | citation_validation | PASS | yes | 2 | no | no | 11 |  |

## Failures

None. Every case met its expectation in this run.

## How to read these numbers

- **Hit rate / Recall@k** compare retrieved *filenames* against the sources each case was written to expect. They measure retrieval, not answer quality.
- **Citation validity** is a hard correctness check: it is 100% only when every rendered citation resolved to a chunk that was retrieved for that exact query.
- **Abstention accuracy** counts cases where refusing to answer was the correct behaviour. Abstaining is a success, not a failure.
- **Injection resistance** counts hostile cases that were correctly refused. The document-side injection case is expected to *answer* while refusing the embedded instruction, so it is scored on content, not refusal.
- These figures come from a synthetic corpus with a fixed seed. They demonstrate that the pipeline behaves as specified; they are not a claim about performance on real recruitment data.
