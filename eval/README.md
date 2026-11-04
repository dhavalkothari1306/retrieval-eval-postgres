Success means: Q2 and Q7 both correct, no new failures on the original 20, and held-out hit@3 not below baseline.
Variants tested: A = RRF, B = action metadata. Nothing else. No tuning of parameters after seeing results.
If a variant fails, I report it as a failure.

Held-out baseline hit@3 is 20/20, so it cannot show improvement and is used only to detect regressions.
Secondary metric added after seeing the baseline: top-1 accuracy (original 19/20, held-out 16/20). Declared before any variant was run.
A second “stress set” was written after seeing the baseline, specifically to be harder, and is reported separately and labeled as post-hoc.

Variant A (RRF): unlikely to fix Q2 or Q41. Keyword search matches on shared tokens, and doc 11 shares none with either query, so the keyword list won’t surface it. Doc 12 shares “workspace” and will rank better in the keyword list, which could even push 12 higher.
Variant B (action metadata): might help, if “add/join a person” lands closer to an action: invite prefix than to doc 12’s text. It could also leave the ranking unchanged or disturb other queries.