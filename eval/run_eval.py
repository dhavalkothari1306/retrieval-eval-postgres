"""
eval/run_eval.py: run a retrieval variant on a query file and report hit@3.

Usage (from repo root):
  python eval/run_eval.py --variant baseline --queries queries.json
  python eval/run_eval.py --variant baseline --queries eval/heldout_queries.json
"""
import argparse
import json
import os

import psycopg2
from sentence_transformers import SentenceTransformer

DB = dict(host="localhost", port=5432, dbname="searchdb",
          user="postgres", password="postgres")

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_conn = psycopg2.connect(**DB)


def semantic_search(query, limit=10):
    """Baseline: pgvector cosine distance, returns ranked doc ids."""
    emb = _model.encode(query).tolist()
    with _conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM documents ORDER BY embedding <=> %s::vector LIMIT %s;",
            (emb, limit),
        )
        return [r[0] for r in cur.fetchall()]


# Variants A (RRF) and B (metadata) get registered here later.
VARIANTS = {"baseline": semantic_search}


def evaluate(search_fn, queries, k=3):
    rows = []
    for q in queries:
        top = search_fn(q["text"])[:k]
        rows.append({
            "id": q["id"],
            "type": q["type"],
            "text": q["text"],
            "correct": q["correct_doc_id"],
            "top3": top,
            "hit": q["correct_doc_id"] in top,
        })
    return rows


def fmt(subset):
    hits = sum(r["hit"] for r in subset)
    return f"{hits}/{len(subset)} ({100 * hits / len(subset):.0f}%)" if subset else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--queries", required=True)
    args = ap.parse_args()

    with open(args.queries, encoding="utf-8") as f:
        queries = json.load(f)

    rows = evaluate(VARIANTS[args.variant], queries)

    print(f"\nvariant={args.variant}  queries={args.queries}\n")
    print(f"{'Q':>3}  {'type':<10}  {'correct':>7}  {'hit':>3}  top3")
    for r in rows:
        print(f"{r['id']:>3}  {r['type']:<10}  {r['correct']:>7}  "
              f"{'Y' if r['hit'] else 'N':>3}  {r['top3']}")

    print(f"\nAll:        {fmt(rows)}")
    print(f"Keyword:    {fmt([r for r in rows if r['type'] == 'keyword'])}")
    print(f"Paraphrase: {fmt([r for r in rows if r['type'] == 'paraphrase'])}")

    os.makedirs("eval/results", exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.queries))[0]
    out = f"eval/results/{args.variant}_{stem}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()