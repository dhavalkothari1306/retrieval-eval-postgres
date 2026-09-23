"""
rerank.py — extends the Post 2 evaluation with a cross-encoder reranking step.

Pipeline:
  1. pgvector retrieves top-10 candidates per query (same as before)
  2. cross-encoder/ms-marco-MiniLM-L-6-v2 rescores each (query, candidate) pair
  3. Top 3 after reranking become the final result

Outputs:
  - hit@3 table: semantic-only vs semantic+reranker, split by query type
  - Q2 deep-dive: raw cross-encoder scores for doc 11 vs doc 12
  - Mean added latency from the reranking step
  - Any regressions (queries correct before reranking that broke after)
"""

import json
import time
import psycopg2
from sentence_transformers import SentenceTransformer, CrossEncoder

DB = dict(host="localhost", port=5432, dbname="searchdb",
          user="postgres", password="postgres")

with open("queries.json") as f:
    queries = json.load(f)

with open("documents.json") as f:
    doc_lookup = {d["id"]: d["text"] for d in json.load(f)}

conn = psycopg2.connect(**DB)
cur = conn.cursor()

print("Loading models...")
bi_encoder    = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Models loaded.\n")

results       = []
rerank_times  = []

for q in queries:
    qid     = q["id"]
    qtext   = q["text"]
    correct = q["correct_doc_id"]
    qtype   = q["type"]

    # Step 1: pgvector top-10 candidates
    embedding = bi_encoder.encode(qtext).tolist()
    cur.execute("""
        SELECT id, text,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT 10;
    """, (embedding, embedding))
    candidates = cur.fetchall()   # [(id, text, score), ...]

    sem_top3   = [r[0] for r in candidates[:3]]
    sem_hit    = correct in sem_top3

    # Step 2: cross-encoder reranking
    t0 = time.perf_counter()
    pairs  = [(qtext, row[1]) for row in candidates]
    scores = cross_encoder.predict(pairs)
    rerank_ms = (time.perf_counter() - t0) * 1000
    rerank_times.append(rerank_ms)

    ranked     = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    rerank_top3 = [r[0][0] for r in ranked[:3]]
    rerank_hit  = correct in rerank_top3

    # Q2 deep-dive: record scores for doc 11 and doc 12 specifically
    q2_scores = {}
    if qid == 2:
        for (doc_id, doc_text, _), score in zip(candidates, scores):
            if doc_id in (11, 12):
                q2_scores[doc_id] = round(float(score), 4)

    results.append({
        "id":          qid,
        "text":        qtext,
        "type":        qtype,
        "correct":     correct,
        "sem_hit":     sem_hit,
        "sem_top1":    sem_top3[0] if sem_top3 else None,
        "rerank_hit":  rerank_hit,
        "rerank_top1": rerank_top3[0] if rerank_top3 else None,
        "rerank_ms":   rerank_ms,
        "q2_scores":   q2_scores,
    })

cur.close()
conn.close()

# ── Results table ──────────────────────────────────────────────────────────
print("="*80)
print(f"{'Q':>2}  {'Type':<10}  {'Correct':>7}  {'SEM':>6}  {'SEM top1':>8}  {'RERANK':>7}  {'Rerank top1':>11}  {'ms':>6}")
print("="*80)
for r in results:
    print(f"{r['id']:>2}  {r['type']:<10}  {r['correct']:>7}  "
          f"{'✓' if r['sem_hit'] else '✗':>6}  {str(r['sem_top1']):>8}  "
          f"{'✓' if r['rerank_hit'] else '✗':>7}  {str(r['rerank_top1']):>11}  "
          f"{r['rerank_ms']:>6.1f}")

# ── Hit@3 summary ──────────────────────────────────────────────────────────
def hit_rate(subset, key):
    hits = sum(1 for r in subset if r[key])
    return hits, len(subset), f"{100*hits/len(subset):.0f}%"

all_r  = results
kw_r   = [r for r in results if r["type"] == "keyword"]
para_r = [r for r in results if r["type"] == "paraphrase"]

print("\n── Hit@3 summary ──────────────────────────────────────────────────────")
print(f"{'Category':<22}  {'Semantic only':>13}  {'Semantic+Reranker':>17}")
print("-"*58)
for label, subset in [("All queries", all_r), ("Keyword queries", kw_r), ("Paraphrase queries", para_r)]:
    sh, st, sp = hit_rate(subset, "sem_hit")
    rh, rt, rp = hit_rate(subset, "rerank_hit")
    print(f"{label:<22}  {sp:>6} ({sh}/{st})  {rp:>10} ({rh}/{rt})")

# ── Q2 deep-dive ───────────────────────────────────────────────────────────
print("\n── Q2 cross-encoder scores (invite vs remove member) ──────────────────")
q2 = next(r for r in results if r["id"] == 2)
print(f"Query: \"{q2['text']}\"")
print(f"Correct doc: 11 (invite teammates)")
if q2["q2_scores"]:
    for doc_id, score in sorted(q2["q2_scores"].items()):
        label = "invite member (CORRECT)" if doc_id == 11 else "remove member (wrong)"
        print(f"  Doc {doc_id} ({label}): score = {score}")
    if 11 in q2["q2_scores"] and 12 in q2["q2_scores"]:
        if q2["q2_scores"][11] > q2["q2_scores"][12]:
            print("  Reranker correctly ranked doc 11 above doc 12.")
        else:
            print("  Reranker still ranked doc 12 above doc 11 — same failure as Post 2.")
else:
    print("  Doc 11 or doc 12 did not appear in top-10 candidates.")
print(f"  Reranker top-1 result: doc {q2['rerank_top1']}")

# ── Regressions ────────────────────────────────────────────────────────────
regressions = [r for r in results if r["sem_hit"] and not r["rerank_hit"]]
print("\n── Regressions (correct before reranking, broken after) ───────────────")
if regressions:
    for r in regressions:
        print(f"  Q{r['id']} [{r['type']}]: \"{r['text']}\"")
        print(f"    Correct: {r['correct']} | Reranker top-1: {r['rerank_top1']}")
else:
    print("  None — reranking did not break any previously correct results.")

# ── Latency ────────────────────────────────────────────────────────────────
mean_ms = sum(rerank_times) / len(rerank_times)
print(f"\n── Reranking latency ──────────────────────────────────────────────────")
print(f"  Mean added latency per query: {mean_ms:.1f}ms")
print(f"  Min: {min(rerank_times):.1f}ms  Max: {max(rerank_times):.1f}ms")
print("\nDone.")