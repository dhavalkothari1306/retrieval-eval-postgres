"""
search.py — runs all 20 labeled queries through:
  1. Keyword search  (tsvector / ts_rank)
  2. Semantic search (pgvector cosine similarity + all-MiniLM-L6-v2)

Prints:
  - Per-query results table showing hit@3 for each method
  - Overall hit@3 rates split by query type (keyword vs paraphrase)
  - Concrete failure examples for each method
"""

import json
import psycopg2
from sentence_transformers import SentenceTransformer

DB = dict(host="localhost", port=5432, dbname="searchdb",
          user="postgres", password="postgres")

with open("queries.json") as f:
    queries = json.load(f)

conn = psycopg2.connect(**DB)
cur = conn.cursor()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

results = []

for q in queries:
    qid        = q["id"]
    qtext      = q["text"]
    correct    = q["correct_doc_id"]
    qtype      = q["type"]

    # --- Keyword search: ts_rank on tsvector ---
    cur.execute("""
        SELECT id, text,
               ts_rank(tsv, plainto_tsquery('english', %s)) AS rank
        FROM documents
        WHERE tsv @@ plainto_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT 3;
    """, (qtext, qtext))
    kw_rows   = cur.fetchall()
    kw_top3   = [r[0] for r in kw_rows]
    kw_hit    = correct in kw_top3
    kw_top1   = kw_top3[0] if kw_top3 else None
    kw_text   = kw_rows[0][1][:80] + "..." if kw_rows else "(no results)"

    # --- Semantic search: cosine similarity on embeddings ---
    embedding = model.encode(qtext).tolist()
    cur.execute("""
        SELECT id, text,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT 3;
    """, (embedding, embedding))
    sem_rows  = cur.fetchall()
    sem_top3  = [r[0] for r in sem_rows]
    sem_hit   = correct in sem_top3
    sem_top1  = sem_top3[0] if sem_top3 else None
    sem_text  = sem_rows[0][1][:80] + "..." if sem_rows else "(no results)"

    results.append({
        "id":       qid,
        "text":     qtext,
        "type":     qtype,
        "correct":  correct,
        "kw_hit":   kw_hit,
        "kw_top1":  kw_top1,
        "kw_text":  kw_text,
        "sem_hit":  sem_hit,
        "sem_top1": sem_top1,
        "sem_text": sem_text,
    })

cur.close()
conn.close()

# ── Results table ──────────────────────────────────────────────────────────
print("\n" + "="*90)
print(f"{'Q':>2}  {'Type':<10}  {'Correct':>7}  {'KW hit':>6}  {'KW top1':>7}  {'SEM hit':>7}  {'SEM top1':>8}")
print("="*90)
for r in results:
    print(f"{r['id']:>2}  {r['type']:<10}  {r['correct']:>7}  "
          f"{'✓' if r['kw_hit'] else '✗':>6}  {str(r['kw_top1']):>7}  "
          f"{'✓' if r['sem_hit'] else '✗':>7}  {str(r['sem_top1']):>8}")

# ── Hit@3 summary ──────────────────────────────────────────────────────────
def hit_rate(subset, method):
    hits = sum(1 for r in subset if r[method])
    return hits, len(subset), f"{100*hits/len(subset):.0f}%"

all_r    = results
kw_r     = [r for r in results if r["type"] == "keyword"]
para_r   = [r for r in results if r["type"] == "paraphrase"]

print("\n── Hit@3 summary ──────────────────────────────────────────────")
print(f"{'Category':<20}  {'Keyword search':>14}  {'Semantic search':>15}")
print("-"*55)
for label, subset in [("All queries", all_r), ("Keyword queries", kw_r), ("Paraphrase queries", para_r)]:
    kh, kt, kp = hit_rate(subset, "kw_hit")
    sh, st, sp = hit_rate(subset, "sem_hit")
    print(f"{label:<20}  {kp:>7} ({kh}/{kt})  {sp:>8} ({sh}/{st})")

# ── Failure examples ───────────────────────────────────────────────────────
print("\n── Where keyword search missed (semantic won) ──────────────────")
for r in results:
    if not r["kw_hit"] and r["sem_hit"]:
        print(f"\nQ{r['id']} [{r['type']}]: \"{r['text']}\"")
        print(f"  Correct doc : {r['correct']}")
        print(f"  KW returned : {r['kw_top1']} — \"{r['kw_text']}\"")
        print(f"  SEM returned: {r['sem_top1']} — \"{r['sem_text']}\"")

print("\n── Where semantic search missed (keyword won) ──────────────────")
for r in results:
    if not r["sem_hit"] and r["kw_hit"]:
        print(f"\nQ{r['id']} [{r['type']}]: \"{r['text']}\"")
        print(f"  Correct doc : {r['correct']}")
        print(f"  SEM returned: {r['sem_top1']} — \"{r['sem_text']}\"")
        print(f"  KW returned : {r['kw_top1']} — \"{r['kw_text']}\"")

print("\n── Where both missed ───────────────────────────────────────────")
for r in results:
    if not r["kw_hit"] and not r["sem_hit"]:
        print(f"\nQ{r['id']} [{r['type']}]: \"{r['text']}\"")
        print(f"  Correct doc : {r['correct']}")
        print(f"  KW returned : {r['kw_top1']} — \"{r['kw_text']}\"")
        print(f"  SEM returned: {r['sem_top1']} — \"{r['sem_text']}\"")

print("\nDone.")
