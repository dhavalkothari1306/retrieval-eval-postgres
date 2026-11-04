Success means: Q2 and Q7 both correct, no new failures on the original 20, and held-out hit@3 not below baseline.
Variants tested: A = RRF, B = action metadata. Nothing else. No tuning of parameters after seeing results.
If a variant fails, I report it as a failure.

Held-out baseline hit@3 is 20/20, so it cannot show improvement and is used only to detect regressions.
Secondary metric added after seeing the baseline: top-1 accuracy (original 19/20, held-out 16/20). Declared before any variant was run.
A second “stress set” was written after seeing the baseline, specifically to be harder, and is reported separately and labeled as post-hoc.

Variant A (RRF): unlikely to fix Q2 or Q41. Keyword search matches on shared tokens, and doc 11 shares none with either query, so the keyword list won’t surface it. Doc 12 shares “workspace” and will rank better in the keyword list, which could even push 12 higher.
Variant B (action metadata): might help, if “add/join a person” lands closer to an action: invite prefix than to doc 12’s text. It could also leave the ranking unchanged or disturb other queries.

Variant A (RRF, k=60, fixed): identical top-3 to baseline on all 52 queries, with no improvements or regressions. Diagnostic: keyword search returned no results for 40/52 queries; in the 12 non-empty cases its top-1 equalled semantic top-1 every time. For Q2 and Q41, the keyword list was empty, so doc 11 could not enter the fused ranking. Conclusion: with AND-semantics keyword search, fusion adds nothing here. An OR-style keyword query was not tested.

Variant B disclosure:

Variant B labels (doc_actions.json) were written after the Q2 failure was known, so B is not a blind test. Prefix format: action: <label>. <doc text>. Labels, prefix wording and model are frozen before the run. Borderline labels: docs 17, 21, 30, 35 (decisions: ...).

Predictions:

Prediction (written before run): B fixes Q2/Q41 with probability under 50%. Reason: doc 11 shares no content words with either query, and a one-word prefix is a weak signal inside a long string. Q2 and Q41 count as one failure because their top-3 results are identical.
Success criterion: Q2 and Q41 both hit@3; no query that was a hit becomes a miss on any set; held-out top-1 not below 16/20.