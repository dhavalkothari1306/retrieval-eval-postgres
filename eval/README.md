# Retrieval fix test: can hybrid fusion or action labels fix a failing query?

A small, reproducible experiment on top of [`retrieval-eval-postgres`](https://github.com/dhavalkothari1306/retrieval-eval-postgres). It tests two cheap fixes for one known retrieval failure and reports what happened, including that neither fixed it.

**Short version:** neither variant fixed the failing query, nothing regressed, and the correct document sat just outside the top 3 under both.

## The failure being tested

Corpus: 40 short help-center documents (synthetic, in `documents.json`). Retrieval: Postgres + pgvector, `all-MiniLM-L6-v2` embeddings, cosine distance.

On the original query set, one query misses at hit@3:

| Query | Text | Correct | Baseline top 3 |
|---|---|---|---|
| Q2 | "What do I need to do to let another person join our shared workspace?" | doc 11 (invite teammates) | `[12, 23, 36]` |

Doc 12 is "remove a member". The query and its correct answer are about adding someone; the top result is about removing someone. A post-hoc stress query (Q41, "How do I add a new person to my workspace?") returns the identical top 3 `[12, 23, 36]`, so **Q2 and Q41 are treated as one failure, not two independent ones**.

## Variants

| Variant | What it does |
|---|---|
| Baseline | pgvector cosine search over document embeddings |
| A: RRF | Reciprocal rank fusion (k=60, not tuned) of Postgres full-text search (`tsvector`, `ts_rank`, `plainto_tsquery`) and the baseline semantic ranking |
| B: action metadata | Each document is re-embedded as `action: <label>. <doc text>` using one label per document from `doc_actions.json`; queries are embedded unchanged |

Nothing else was tested.

## Query sets

| Set | Size | File | Notes |
|---|---|---|---|
| Original | 20 | `queries.json` | From the earlier posts. Q2 is the known miss |
| Held-out | 20 | `eval/heldout_queries.json` | Written and frozen before any variant was run. Baseline already scores 20/20, so it can only detect regressions |
| Stress | 12 | `eval/stress_queries.json` | Written **after** seeing the baseline, to be harder (opposite-action and near-duplicate documents). Post-hoc |

Metrics: hit@3 (the correct document is in the top 3) and top-1 (the correct document is ranked first). Top-1 was added after seeing the baseline but before any variant was run.

## Pre-registered plan

Committed before running any variant (see git history):

- Success: Q2 correct, no new failures on the original 20, held-out hit@3 not below baseline.
- Variants tested: A and B only. No parameter tuning after seeing results. A failing variant is reported as failed.
- The held-out set is used only to detect regressions, because the baseline is already at its ceiling.

**Predictions, written before the runs:**

- **A (RRF):** unlikely to fix Q2 or Q41. Keyword search needs shared tokens, and doc 11 shares none with either query. Doc 12 shares "workspace", so fusion might even push it higher.
- **B (metadata):** under 50% chance of fixing Q2/Q41. A one-word prefix is a weak signal inside a long string, and doc 11 still shares no content words with the queries. It could also leave rankings unchanged or disturb other queries.

**Extended success criterion (applied after the stress set was added):** Q2 and Q41 both hit@3; no query that was a hit becomes a miss on any set; held-out top-1 not below 16/20.

## Results

| Variant | Original hit@3 / top-1 | Held-out hit@3 / top-1 | Stress hit@3 / top-1 | Q2 / Q41 |
|---|---|---|---|---|
| Baseline | 19/20, 19/20 | 20/20, 16/20 | 11/12, 10/12 | miss / miss |
| A: RRF | 19/20, 19/20 | 20/20, 16/20 | 11/12, 10/12 | miss / miss |
| B: metadata | 19/20, 19/20 | 20/20, 18/20 | 11/12, 10/12 | miss / miss |

**Both variants fail the primary criterion.** No hit became a miss on any set.

### Variant A (RRF)

Top-3 results are identical to baseline on all 52 queries.

Diagnostic (`eval/diagnose_keyword.py`): the keyword search returned no results for 40 of 52 queries. In the 12 non-empty cases, its top-1 equalled the semantic top-1 every time. For Q2 and Q41 the keyword list was empty, so doc 11 had no route into the fused ranking. With AND-semantics keyword matching, fusion added nothing on this corpus.

Not tested: an OR-style keyword query. That would be a new variant added after seeing results and would need to be reported as exploratory.

### Variant B (action metadata)

- Top-3 changed on 32 of 52 queries (13 original, 12 held-out, 7 stress), mostly in lower positions.
- Top-1 changed on only 2 queries, both in the held-out set (Q23 and Q36, each moving to rank 1). Held-out top-1 went from 16/20 to 18/20. No mechanism was tested. Treat this as a small side effect, not a finding: n=2, labels were written knowing about Q2, and top-1 was added after seeing the baseline.
- The "remove" queries (Q26, Q42) were unchanged and still correct.

### Where is doc 11?

Post-hoc diagnostic (`eval/diagnose_doc11.py`), rank of doc 11 out of 40:

| Query | Baseline | Variant B |
|---|---|---|
| Q2 | 4 | 4 |
| Q41 | 6 | 6 |

Variant B did not change doc 11's rank on either query. Doc 11 is near the top but below three other documents. **Why those documents outrank it was not tested.** Q2 and Q41 both contain "workspace", as do several documents above it, which is a plausible explanation but only a hypothesis.

## Disclosures and limitations

- Corpus is 40 synthetic documents; query sets total 52. One query is worth 2 to 5 percentage points, so none of the differences here are statistically meaningful.
- Variant B labels (`doc_actions.json`) were written **after** the Q2 failure was known. B is not a blind test. Labels, prefix wording and model were frozen before it was run. Borderline labels: docs 17, 21, 30, 35. <!-- TODO: add the decision made for each of these four -->
- The stress set was written after the baseline was seen, and one of its queries (Q41) targets the known failure.
- The held-out set is at 100% baseline hit@3, so it cannot show improvement.
- Only one keyword configuration (AND semantics, `plainto_tsquery`) and one embedding model were tested.
- Keyword search SQL adds a deterministic `id` tiebreak that the original Post 2 query did not have. It did not change any result.
- Latency and cost were not measured for either variant.
- Results are from a single run per variant. Search is deterministic here, but this was not repeated across machines or library versions.

## Reproduce

Requires Docker with `pgvector/pgvector:pg16`, a `searchdb` database, and a `documents` table with `embedding vector(384)` and a `tsv` column, as set up in the earlier posts. <!-- TODO: link the setup/seed script -->

```bash
pip install psycopg2-binary sentence-transformers

# Variant B only: add a second embedding column (original column is untouched)
python eval/build_meta_embeddings.py

# Run each variant on each set
python eval/run_eval.py --variant baseline --queries queries.json
python eval/run_eval.py --variant rrf      --queries queries.json
python eval/run_eval.py --variant meta     --queries queries.json
# repeat with eval/heldout_queries.json and eval/stress_queries.json

# Per-query diff between two runs (prints only queries whose top 3 changed)
python eval/compare.py eval/results/baseline_queries.json eval/results/meta_queries.json

# Diagnostics
python eval/diagnose_keyword.py
python eval/diagnose_doc11.py
```

Run from the repository root. Per-query results are saved to `eval/results/`.

## Files

| Path | Purpose |
|---|---|
| `eval/run_eval.py` | Evaluation harness and the three variants |
| `eval/build_meta_embeddings.py` | Builds the `embedding_meta` column for Variant B |
| `eval/doc_actions.json` | One action label per document (Variant B input) |
| `eval/heldout_queries.json` | Frozen held-out queries |
| `eval/stress_queries.json` | Post-hoc stress queries |
| `eval/compare.py` | Per-query diff between two result files |
| `eval/diagnose_keyword.py` | Counts how often keyword search contributes to RRF |
| `eval/diagnose_doc11.py` | Reports the rank of doc 11 for Q2 and Q41 |
| `eval/results/` | Per-query results for every variant and set |

## Deviations from the plan

- The pre-registered criterion originally also named Q7. Q7 is a TLS query that the cross-encoder reranker broke in the earlier post; under the baseline used here it is already a hit, so it is not a target. Q41 was added to the criterion when the stress set was written.
- Top-1 and the stress set were added after the baseline was seen. Both are labeled above.

## Possible next steps (not done)

- An OR-style keyword query for the keyword leg of RRF.
- Rewording doc 11 to test whether the miss comes from the document's wording. This edits the corpus to fit the failing query, so it would be a diagnosis, not a fix.
- Measuring hit@5 to see how many misses are near misses.
