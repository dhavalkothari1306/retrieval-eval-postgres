"""Post-hoc diagnostic: how often does keyword search contribute anything to RRF?"""
import json
from run_eval import keyword_search, semantic_search

SETS = ["queries.json", "eval/heldout_queries.json", "eval/stress_queries.json"]

empty = nonempty = agree_top1 = 0
for path in SETS:
    with open(path, encoding="utf-8") as f:
        queries = json.load(f)
    for q in queries:
        kw = keyword_search(q["text"])
        sem = semantic_search(q["text"])[:3]
        if not kw:
            empty += 1
        else:
            nonempty += 1
            agree_top1 += kw[0] == sem[0]
        if q["id"] in (2, 41):
            print(f"Q{q['id']}: keyword={kw} semantic_top3={sem} doc11_in_keyword={11 in kw}")

print(f"\nkeyword list empty: {empty}/{empty + nonempty}")
print(f"keyword non-empty: {nonempty}; of those, keyword top-1 == semantic top-1: {agree_top1}")