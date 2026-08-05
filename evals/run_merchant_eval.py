"""Merchant-resolution eval — §9.2/§9.3.

Reports three things, because the first one alone is misleading:

  accuracy   per-row, EXACT canonical match against ground truth
  splits     one true merchant mapped onto several canonical entries
  tiers      % resolved without an LLM call

An earlier version of this scorer used prefix matching, so "Shell Oil" counted
as a hit against "Shell" and a 5-way merchant split scored 95%. Splits are
invisible per-row and fatal in aggregate — a query for one merchant silently
returns a fraction of its true total — so they get their own number.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # evals/ is not a package

from ledgerlens.db import DB_PATH, connect, init_db
from ledgerlens.ingest import ingest_file
from ledgerlens.ingest.categorize import categorize
from ledgerlens.ingest.merchants import resolve
from ledgerlens.seed import seed

LABELS = Path(__file__).parent / "labeled_categories.csv"
SYNTHETIC = Path(__file__).parent.parent / "data" / "synthetic" / "transactions.csv"


def build(reset: bool = True, ablate_rules: bool = False) -> dict:
    """Ingest, resolve and categorize the whole synthetic ledger.

    `ablate_rules` deletes the seeded regexes before categorizing, which is the
    only honest way to read the headline accuracy. The 14 rules in seed.py were
    written *knowing* this catalog, so scoring them against it measures how well
    the rules were written, not how well the pipeline generalizes. Removing them
    forces every merchant through the LLM tier and lets the learned write-back
    propagate that decision — including its mistakes, which is exactly the
    failure mode a new statement would hit.
    """
    if reset:
        DB_PATH.unlink(missing_ok=True)
        init_db()

    started = time.time()
    tiers: collections.Counter[int] = collections.Counter()
    cat_tiers: collections.Counter[int] = collections.Counter()
    review_queue = 0

    with connect() as conn:
        ingest_file(conn, SYNTHETIC)
        seed(conn)
        if ablate_rules:
            conn.execute("DELETE FROM category_rules")
            conn.commit()
        rows = conn.execute(
            "SELECT id, raw_descriptor, type FROM transactions ORDER BY id"
        ).fetchall()

        for tid, descriptor, txn_type in rows:
            m = resolve(conn, descriptor)
            tiers[m.tier] += 1

            c = categorize(conn, descriptor, m.merchant_id, txn_type)
            cat_tiers[c.tier] += 1
            review_queue += c.needs_review

            conn.execute(
                """UPDATE transactions
                   SET merchant_id = ?, category_id = ?, categorized_by = ?, confidence = ?
                   WHERE id = ?""",
                (m.merchant_id, c.category_id, c.categorized_by, c.confidence, tid),
            )
        conn.commit()

    total = sum(tiers.values())
    return {
        "transactions": total,
        "tier1_alias": tiers[1],
        "tier2_rule": tiers[2],
        "tier3_llm": tiers[3],
        "no_llm_pct": (tiers[1] + tiers[2]) / total * 100,
        "cat_user": cat_tiers[1],
        "cat_learned": cat_tiers[2],
        "cat_rule": cat_tiers[3],
        "cat_llm": cat_tiers[4],
        "cat_no_llm_pct": (total - cat_tiers[4]) / total * 100,
        "review_queue": review_queue,
        "seconds": time.time() - started,
    }


def score_categories() -> dict:
    """§9.2 — accuracy overall and per resolution tier."""
    labels = list(csv.DictReader(LABELS.open()))
    by_tier: dict[str, list[bool]] = collections.defaultdict(list)
    confusion: collections.Counter[tuple[str, str]] = collections.Counter()
    hits = 0

    with connect() as conn:
        names = {cid: name for cid, name in conn.execute("SELECT id, name FROM categories")}
        for row in labels:
            got = conn.execute(
                """SELECT category_id, categorized_by FROM transactions
                   WHERE raw_descriptor = ? LIMIT 1""",
                (row["raw_descriptor"],),
            ).fetchone()
            if not got:
                continue
            predicted = names.get(got[0], "uncategorized")
            expected = row["expected_category"]
            ok = predicted == expected
            hits += ok
            by_tier[got[1] or "none"].append(ok)
            if not ok:
                confusion[(expected, predicted)] += 1

    return {
        "labels": len(labels),
        "accuracy_pct": hits / len(labels) * 100,
        "per_tier": {
            tier: {"n": len(v), "accuracy_pct": sum(v) / len(v) * 100}
            for tier, v in sorted(by_tier.items())
        },
        "confusion": confusion.most_common(8),
    }


def score() -> dict:
    labels = list(csv.DictReader(LABELS.open()))
    true_to_pred: dict[str, set[str]] = collections.defaultdict(set)
    pred_to_true: dict[str, set[str]] = collections.defaultdict(set)
    hits = 0

    with connect() as conn:
        for row in labels:
            got = conn.execute(
                """SELECT m.canonical_name FROM transactions t
                   JOIN merchants m ON m.id = t.merchant_id
                   WHERE t.raw_descriptor = ? LIMIT 1""",
                (row["raw_descriptor"],),
            ).fetchone()
            predicted = got[0] if got else None
            expected = row["expected_merchant"]
            true_to_pred[expected].add(predicted)
            pred_to_true[predicted].add(expected)
            if predicted == expected:
                hits += 1

        merchant_count = conn.execute("SELECT COUNT(*) FROM merchants").fetchone()[0]

    splits = {k: sorted(v) for k, v in true_to_pred.items() if len(v) > 1}
    merges = {k: sorted(v) for k, v in pred_to_true.items() if len(v) > 1}
    renames = {k: next(iter(v)) for k, v in true_to_pred.items()
               if len(v) == 1 and next(iter(v)) != k}

    # Clustering is what determines whether aggregates are right. A split
    # undercounts a merchant's total; a merge fuses two businesses. Exact string
    # match is a weaker, cosmetic measure — it penalizes "AMC" vs "AMC Theatres",
    # which changes no total anywhere.
    clustered = len(true_to_pred) - len(splits) - sum(len(v) for v in merges.values())
    return {
        "labels": len(labels),
        "true_merchants": len(true_to_pred),
        "exact_accuracy_pct": hits / len(labels) * 100,
        "cluster_accuracy_pct": clustered / len(true_to_pred) * 100,
        "merchants_created": merchant_count,
        "split_merchants": splits,
        "merged_merchants": merges,
        "renames": renames,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate-rules", action="store_true",
                    help="drop the seeded regexes; measures the pipeline, not the rules")
    args = ap.parse_args()

    build_stats = build(ablate_rules=args.ablate_rules)
    if args.ablate_rules:
        print("!!! category_rules ablated — LLM tier carries categorization\n")
    print("--- §5.3 tier hit rate ---")
    for k in ("transactions", "tier1_alias", "tier2_rule", "tier3_llm"):
        print(f"  {k:16} {build_stats[k]}")
    print(f"  {'no_llm_pct':16} {build_stats['no_llm_pct']:.1f}%")
    print(f"  {'seconds':16} {build_stats['seconds']:.1f}")

    s = score()
    print("\n--- §9.2 merchant accuracy ---")
    print(f"  cluster accuracy  {s['cluster_accuracy_pct']:.1f}%   "
          f"({s['true_merchants']} true merchants, 0 splits/merges = aggregates correct)")
    print(f"  exact string match {s['exact_accuracy_pct']:.1f}%  ({s['labels']} labels)")
    print(f"  merchants created  {s['merchants_created']}")

    for label, data in (("SPLIT", s["split_merchants"]), ("MERGED", s["merged_merchants"])):
        if data:
            print(f"\n  {label} ({len(data)}):")
            for k, v in data.items():
                print(f"    {k:24} -> {v}")

    if s["renames"]:
        print(f"\n  naming differences only ({len(s['renames'])}, no effect on totals):")
        for true_name, predicted in s["renames"].items():
            print(f"    {true_name:24} -> {predicted}")

    print("\n--- §5.4 categorization tier hit rate ---")
    for k in ("cat_user", "cat_learned", "cat_rule", "cat_llm"):
        print(f"  {k:16} {build_stats[k]}")
    print(f"  {'no_llm_pct':16} {build_stats['cat_no_llm_pct']:.1f}%")
    print(f"  {'review_queue':16} {build_stats['review_queue']}")

    c = score_categories()
    print("\n--- §9.2 categorization accuracy ---")
    print(f"  overall           {c['accuracy_pct']:.1f}%  ({c['labels']} labels)")
    for tier, stats in c["per_tier"].items():
        print(f"    {tier:14} {stats['accuracy_pct']:5.1f}%  (n={stats['n']})")
    if c["confusion"]:
        print("\n  most common confusions (expected -> predicted):")
        for (exp, pred), n in c["confusion"]:
            print(f"    {exp:16} -> {pred:16} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
