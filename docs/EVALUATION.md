# Evaluation

The full measurement write-up for [LedgerLens](../README.md) — every number on the front page,
how it was taken, and what it does and does not support.

Model: **`openai/gpt-oss-120b`** on Groq. Data: the seeded synthetic ledger (832 transactions,
12 months, 22 merchants).

Two caveats belong at the top rather than in a footnote:

**Run-to-run variance is large, and every figure here is one run.** Routing accuracy read 50.0%
and then 44.4% across two runs with no routing code changed between them. Treat single-digit
differences as noise; the section on model selection below is a worked example of what that
costs when it is ignored.

**Everything marked *historical* was measured on `llama-3.3-70b-versatile`, which Groq has since
retired — the model id now returns a 404.** Those numbers were real when taken and cannot be
reproduced today, so they are labelled rather than deleted or quietly restated.

## Query accuracy — 53 golden queries *(historical)*

| Metric | Value |
|---|---|
| Execution accuracy, answerable questions | **80.4%** (37/46) |
| Refusal accuracy, unanswerable questions | **100%** (7/7) |
| Whole golden set | **83.0%** (44/53) |
| SQL validity, first attempt | 100% |
| SQL validity, after repair | 100% |
| Queries rescued by the repair loop | 0 |
| Median latency | 4.63 s |
| p90 latency | 5.63 s |
| Median tokens per query | 958 in / 56.5 out |
| Cost per query | ~$0.0006 |

```
python evals/run_golden_eval.py
```

The repair loop rescued nothing on this run, and that is worth stating plainly rather than
hiding: with the 70B, first-attempt SQL validity is already 100%, so there is nothing left for it
to repair. It earned its place against the 8B (95.7% first-attempt) and is now insurance.

This table measures the SQL tool, not the agent: `run_golden_eval.py` calls `run_query()`
directly, so the planner never ran and the graph was never built. That is a real number for
text-to-SQL and it is not a number for a three-tool agent — which is what the routing table below
exists to measure, and why it reads so much lower.

## Routing — 18 questions through the whole graph

The table above measures the SQL tool. This one measures the agent: every question goes through
`ask()`, so the planner runs, the graph is built, and `semantic_tool` / `anomaly_tool` execute
under evaluation for the first time.

```
python evals/run_routing_eval.py
```

| Run | Routing accuracy | Answer accuracy | Verified **and** wrong |
|---|---|---|---|
| Baseline | 38.9% | 38.9% | 7 / 18 |
| Vocabulary in the planner prompt | 50.0% | 33.3% | 11 / 18 |
| ...plus concept scoping, tool chaining | 44.4% | 33.3% | 11 / 18 |
| ...plus ids withheld, honest empty-retrieval wording † | 38.5% | 46.2% | 2 / 13 |

† Truncated: the daily Groq token budget ran out with 5 of the 18 queries left, so that row is
scored over the 13 that reached the model and is **not** comparable with the rows above it. The
harness now excludes 429s from its denominators instead of averaging them in as zeros — before
that fix this run reported 27.8%, which was a quota problem wearing a quality problem's clothes.

**What this table honestly supports, and what it does not.** Routing spans 38.5–50.0% across
four runs; that spread is wider than any single change measured, so no row here demonstrates an
improvement in the headline. The per-tool number is firmer: semantic questions routed to
`semantic` went from 2 of 10 to 4–6 of 10 and stayed there across every run after the prompt
change. `verified_but_wrong` falling from 7 to 2 is the other real movement. **And the last code
change on that list is unmeasured** — the quota was gone before it could be run.

Four things this exercise found that the SQL-only harness structurally could not:

- **A semantic-only plan could never answer.** `semantic_tool` returns ids and an explicitly
  empty `rows`, and the graph routed such a plan straight to `answer`, which then reported "no
  matching transactions" over a retrieval that had succeeded. A unit test asserted this dead end
  *as correct behaviour* and had passed for the life of the project.
- **Retrieval was capping every total at 20 rows.** SQL was scoped to the retrieved ids, so
  "anything medical-looking" — 52 health transactions — summed twenty of them and returned
  -805.93 against a true -1874.81. It passed verification, because every figure genuinely came
  from a retrieved row.
- **The instruction lost to the affordance.** Passing the ids alongside "prefer the category or
  merchant" did not work; the model used the ids anyway. Removing them from the prompt is the fix.
- **A plan naming two tools ran one.** The anomaly branch went straight to `answer`, so the sql
  half of a two-step plan was dropped silently.

The verifier's precision is the uncomfortable number. It checks **provenance, not relevance**:
asked for normal travel spend *ignoring the spike*, the agent averages including the spike and
passes, because the figure did come from a row. Fabrication is caught; wrongness is not.

## Verifier — inject wrong numbers, confirm rejection

| Metric | Value |
|---|---|
| Catch rate on corrupted answers | **100%** (161/161) |
| False rejection rate on correct answers | **0%** (0/41) |

```
python evals/run_verifier_eval.py
```

The one number here that is current rather than historical, because this suite makes no API
calls — the verifier is a pure function and the answer keys come out of the ledger. It had
stopped running entirely: the anomaly questions carry a `reference_fn` instead of
`reference_sql`, and this loop assumed every entry had SQL, so it had raised `KeyError` since
the day those questions were added.

Both numbers are reported together on purpose. A verifier that rejects everything scores 100%
catch and is worthless; one that rejects correct answers is worse than absent, because it teaches
the reader to ignore the verdict. Corruptions are multiplicative (±20%, ±35%, ×2, ×10) and any
that land inside rounding tolerance are discarded as invalid trials rather than counted as misses.
The verifier runs entirely on retrieved rows — no LLM, no network, fully deterministic.

## Merchant resolution — 200 hand-labeled descriptors *(historical)*

| Metric | Value |
|---|---|
| Cluster accuracy (0 splits, 0 merges across the 18 labeled merchants) | **100%** |
| Exact canonical-string match | 84.5% |
| Resolved without an LLM call | **96.9%** |

```
python evals/run_merchant_eval.py
```

**This one rebuilds `ledger.db`.** It re-ingests and re-resolves the whole synthetic ledger, and
tier-3 resolution is an LLM call, so it re-labels the merchants rather than reproducing them —
which is why these figures have not simply been re-taken on the current model. Re-running it
invalidates the golden set's answer key until the reference queries are checked against the new
names.

Cluster accuracy is the number that matters and exact match is the cosmetic one. A *split* —
one real merchant landing on two canonical entries — is invisible per-row and fatal in aggregate,
because every query for that merchant silently returns a fraction of its true total. Exact match
penalizes `AMC` vs `AMC Theatres`, which changes no total anywhere. An earlier version of this
scorer used prefix matching and gave a 5-way merchant split a 95% score; that is why splits now
get their own metric.

## Categorization — the same 200 labels *(historical)*

| Metric | Rules on | Rules ablated |
|---|---|---|
| Accuracy | 100% | 100% |
| Resolved without an LLM call | 100% | 97.7% |
| Transactions reaching the LLM tier | 0 | 19 |

```
python evals/run_merchant_eval.py --ablate-rules
```

**Read both columns, and read the caveat.** The 14 seeded regexes were written knowing this
merchant catalog, so scoring them against it measures how well the rules were written, not how
well the pipeline generalizes — which is why `--ablate-rules` exists. Deleting them forces every
unseen merchant through the LLM tier and lets the learned write-back propagate that decision,
mistakes included. Two further limits: the 200 labels come from the same generator as the
transactions, so neither column is a generalization estimate; and the ablated run is a single
run I was unable to repeat before exhausting the daily API quota. An earlier ablation on
`llama-3.1-8b-instant` scored 97.0%.

## Model selection *(historical)*

`llama-3.3-70b-versatile` over `llama-3.1-8b-instant`, measured on the golden set at the time of
the switch: **67.4% vs 52.2%** execution accuracy, **100% vs 95.7%** first-attempt SQL validity,
**7.6 s vs 13.7 s** median latency — the larger model is faster end-to-end because it needs no
repair round trips.

The honest reading: the 8B's run-to-run noise floor is 2.2 points, but the 70B's own is 10.9
points, so a +15.2 point gap is suggestive rather than decisive. Two prompt rewrites on the 8B
moved accuracy by −4.4 and −2.2 points, both inside its noise — measuring the noise floor first
is what stopped me from shipping either as an improvement.

## Where the 9 remaining failures come from *(historical)*

| Cause | Count |
|---|---|
| Sign convention (`amount` is negative for outflow) | 4 |
| Aggregation level (per-transaction vs per-month) | 2 |
| Wrong `type` filter (fees are `type='fee'`, not `'purchase'`) | 1 |
| Wrong date range | 1 |
| Correct row, wrong column returned | 1 |

The dominant failure class is a single schema convention. `ORDER BY typical_amount DESC` returns
the *cheapest* recurring charge; `SUM(income) - SUM(purchase)` *adds* spending to income. These
are the failures the safety nets cannot catch: the SQL is valid, it executes, it returns a
number, and the verifier confirms that number came from a row — because it did. **Safety nets
catch failure, not wrongness.** The repair loop only sees exceptions, confidence thresholds only
see uncertainty, and a confidently wrong query looks exactly like a right one from the outside.
Fixing this class means teaching the sign convention in the prompt, not adding another guard.

---

## Reproducibility

`--rebuild` is opt-in because it re-ingests the synthetic ledger from scratch, which deletes
`ledger.db`. That is correct for a benchmark and destructive for anyone with real statements
loaded.

**The generator is seeded; the ledger is not reproducible.** Transactions come out identical
every time, but tier-3 merchant resolution calls a model, so the `merchants` table does not. One
rebuild resolved a descriptor to `Delta` and the next to `Delta Air Lines`, which silently
emptied every reference query matching `canonical_name = 'Delta'` — a drifted answer key reads
as a model regression. The golden set matches merchants by prefix (`LIKE 'Delta%'`) for exactly
this reason. Rebuilding is a re-labelling, not a replay.
