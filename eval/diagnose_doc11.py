"""Post-hoc diagnostic: where does doc 11 rank for Q2 and Q41, baseline vs meta?"""
import json
from run_eval import semantic_search, semantic_search_meta

with open("queries.json", encoding="utf-8") as f:
    q2 = next(q for q in json.load(f) if q["id"] == 2)
with open("eval/stress_queries.json", encoding="utf-8") as f:
    q41 = next(q for q in json.load(f) if q["id"] == 41)

for q in (q2, q41):
    for name, fn in (("baseline", semantic_search), ("meta", semantic_search_meta)):
        ranking = fn(q["text"], limit=40)
        print(f"Q{q['id']} {name}: doc 11 rank = {ranking.index(11) + 1}/40, top5 = {ranking[:5]}")