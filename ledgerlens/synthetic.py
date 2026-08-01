"""Synthetic statement generator — §10.

Every screenshot, test, and demo uses synthetic data. This produces ~800
plausible transactions across 12 months, plus the ground truth that Week 1's
gate (§11) and the eval harness (§9) measure against.

Two properties matter more than realism:

1. **Seeded.** Same seed, same rows, byte for byte. Evals that assert on
   specific values are worthless if the data drifts underneath them.
2. **Adversarially messy descriptors.** Clean descriptors would make §5.3's
   tiered merchant resolution look good for the wrong reason. These carry the
   real-world noise the rule tier is written to strip: processor prefixes
   (SQ *, TST*, PAYPAL *), store numbers, city/state tails, embedded dates.

The generator also injects the exact anomalies §7's detectors look for, and
writes a manifest recording where it put them — so a detector's recall is
measurable rather than eyeballed.

    python -m ledgerlens.synthetic
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"

ANCHOR = date(2026, 8, 1)   # fixed, not date.today() — reproducibility
MONTHS = 12
SEED = 20260801

CITIES = [
    ("AUSTIN", "TX"), ("BOSTON", "MA"), ("SEATTLE", "WA"),
    ("DENVER", "CO"), ("CHICAGO", "IL"), ("PORTLAND", "OR"),
]


@dataclass
class MerchantSpec:
    canonical: str
    category: str
    templates: list[str]
    amount: tuple[float, float]
    kind: str                        # recurring | frequent | occasional
    cadence_days: int | None = None  # recurring only
    per_month: tuple[int, int] = (0, 0)  # frequent/occasional draw count
    txn_type: str = "purchase"
    essential: bool = False


# Descriptor templates deliberately include the noise §5.3 tier 2 must strip.
CATALOG: list[MerchantSpec] = [
    # --- recurring subscriptions (§5.5 series detection) ---
    MerchantSpec("Netflix", "subscriptions", ["NETFLIX.COM {city} {state}", "NETFLIX *MONTHLY"],
                 (15.49, 15.49), "recurring", cadence_days=30),
    MerchantSpec("Spotify", "subscriptions", ["SPOTIFY P{store:07d} NEW YORK NY", "PAYPAL *SPOTIFY USA"],
                 (11.99, 11.99), "recurring", cadence_days=30),
    MerchantSpec("Planet Fitness", "health", ["PLANET FIT #{store:04d} {city} {state}"],
                 (24.99, 24.99), "recurring", cadence_days=30),
    MerchantSpec("Adobe", "subscriptions", ["ADOBE *CREATIVE CLD", "ADOBE INC. 408-536-6000 CA"],
                 (22.99, 22.99), "recurring", cadence_days=30),
    MerchantSpec("Comcast", "utilities", ["COMCAST CABLE COMM {city}", "COMCAST *XFINITY {date_mmdd}"],
                 (79.00, 79.00), "recurring", cadence_days=30, essential=True),
    MerchantSpec("Greenline Apartments", "rent", ["GREENLINE APTS RENT {date_mmdd}"],
                 (1450.00, 1450.00), "recurring", cadence_days=30, essential=True),

    # --- frequent everyday spend ---
    MerchantSpec("Trader Joe's", "groceries",
                 ["TRADER JOE'S #{store:03d} {city} {state}", "TRADER JOES #{store:03d} QPS {city}"],
                 (18.40, 96.75), "frequent", per_month=(6, 10), essential=True),
    MerchantSpec("Whole Foods", "groceries",
                 ["WHOLEFDS {city} #{store:05d}", "WHOLE FOODS MKT {city} {state} {date_mmdd}"],
                 (22.10, 134.20), "frequent", per_month=(3, 6), essential=True),
    MerchantSpec("Blue Bottle Coffee", "coffee",
                 ["SQ *BLUE BOTTLE COFFEE {city}", "SQ *BLUE BOTTLE {city} {state}"],
                 (4.25, 9.80), "frequent", per_month=(8, 15)),
    MerchantSpec("Chipotle", "dining",
                 ["CHIPOTLE {store:04d} {city} {state}", "TST* CHIPOTLE - {city}"],
                 (11.20, 24.60), "frequent", per_month=(5, 9)),
    MerchantSpec("Uber", "transport",
                 ["UBER *TRIP {date_mmdd}", "UBER TRIP HELP.UBER.COM"],
                 (7.30, 42.15), "frequent", per_month=(5, 10)),
    MerchantSpec("Shell", "transport",
                 ["SHELL OIL {store:08d} {city} {state}", "SHELL SERVICE STATION #{store:04d}"],
                 (28.00, 61.40), "frequent", per_month=(2, 5)),

    # --- occasional ---
    MerchantSpec("Amazon", "shopping",
                 ["AMZN Mktp US*{ref}", "AMAZON.COM*{ref} AMZN.COM/BILL"],
                 (9.99, 189.00), "occasional", per_month=(3, 8)),
    MerchantSpec("Target", "shopping",
                 ["TARGET T-{store:04d} {city} {state}", "TARGET.COM * {ref}"],
                 (15.30, 142.80), "occasional", per_month=(2, 4)),
    MerchantSpec("CVS Pharmacy", "health",
                 ["CVS/PHARMACY #{store:05d} {city}", "CVS {store:05d} {city} {state}"],
                 (6.40, 78.90), "occasional", per_month=(1, 4)),
    MerchantSpec("AMC Theatres", "entertainment",
                 ["AMC ONLINE {ref}", "AMC #{store:04d} {city} {state}"],
                 (14.50, 38.00), "occasional", per_month=(0, 3)),
    MerchantSpec("Delta Air Lines", "travel",
                 ["DELTA AIR {ref} ATLANTA GA", "DELTA AIR LINES {date_mmdd}"],
                 (180.00, 640.00), "occasional", per_month=(0, 1)),
    MerchantSpec("Marriott", "travel",
                 ["MARRIOTT {city} {state}", "MARRIOTT HTL {store:04d} {city}"],
                 (145.00, 420.00), "occasional", per_month=(0, 1)),

    # --- non-purchase types ---
    MerchantSpec("Acme Corp Payroll", "income", ["ACME CORP DIRECT DEP PPD ID:{ref}"],
                 (2350.00, 2350.00), "recurring", cadence_days=14, txn_type="income"),
    MerchantSpec("Savings Transfer", "transfer", ["ONLINE TRANSFER TO SAV {ref}"],
                 (400.00, 400.00), "recurring", cadence_days=30, txn_type="transfer"),
    MerchantSpec("Bank Fee", "fees", ["MONTHLY MAINTENANCE FEE", "ATM FEE {city} {state}"],
                 (2.50, 3.50), "occasional", per_month=(0, 1), txn_type="fee"),
    # Returns land as inflow against the original merchant — analytics queries
    # filtering type='purchase' must not double-count these.
    MerchantSpec("Amazon", "shopping", ["AMZN Mktp US*{ref} REFUND", "AMAZON.COM*{ref} RETURN"],
                 (12.00, 85.00), "occasional", per_month=(0, 1), txn_type="refund"),
]

# Appears only in the final month — §7 new_merchant detector.
NEW_MERCHANT = MerchantSpec(
    "Peloton", "health", ["PELOTON *MEMBERSHIP", "PELOTON INTERACTIVE NY"],
    (44.00, 44.00), "recurring", cadence_days=30,
)


@dataclass
class Manifest:
    """Ground truth for the injected anomalies, so §7 recall is measurable."""
    price_hike: dict = field(default_factory=dict)
    duplicate: dict = field(default_factory=dict)
    spend_spike: dict = field(default_factory=dict)
    new_merchant: dict = field(default_factory=dict)


def _month_starts(anchor: date, months: int) -> list[date]:
    """First day of each month in the window, oldest first."""
    starts, y, m = [], anchor.year, anchor.month
    for _ in range(months):
        m -= 1
        if m == 0:
            m, y = 12, y - 1
        starts.append(date(y, m, 1))
    return sorted(starts)


def _days_in_month(d: date) -> int:
    nxt = date(d.year + (d.month == 12), (d.month % 12) + 1, 1)
    return (nxt - d).days


def _descriptor(rng: random.Random, spec: MerchantSpec, when: date) -> str:
    city, state = rng.choice(CITIES)
    return rng.choice(spec.templates).format(
        city=city,
        state=state,
        store=rng.randint(1, 99999),
        ref="".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789", k=9)),
        date_mmdd=f"{when.month:02d}{when.day:02d}",
    )


def _amount(rng: random.Random, spec: MerchantSpec) -> float:
    lo, hi = spec.amount
    value = lo if lo == hi else rng.uniform(lo, hi)
    # Non-negotiable convention: negative = outflow. Income and refunds are inflow.
    sign = 1 if spec.txn_type in ("income", "refund") else -1
    return round(sign * value, 2)


def generate(seed: int = SEED, anchor: date = ANCHOR, months: int = MONTHS) -> tuple[list[dict], Manifest]:
    rng = random.Random(seed)
    starts = _month_starts(anchor, months)
    rows: list[dict] = []
    manifest = Manifest()

    def emit(spec: MerchantSpec, when: date, amount: float | None = None) -> dict:
        row = {
            "account_id": 1,
            "posted_date": when.isoformat(),
            "amount": amount if amount is not None else _amount(rng, spec),
            "currency": "USD",
            "raw_descriptor": _descriptor(rng, spec, when),
            "type": spec.txn_type,
            "source_file": f"synthetic_{when.year}_{when.month:02d}.csv",
            "true_merchant": spec.canonical,
            "true_category": spec.category,
        }
        rows.append(row)
        return row

    # --- recurring series ---
    hike_spec = next(s for s in CATALOG if s.canonical == "Spotify")
    for spec in CATALOG:
        if spec.kind != "recurring":
            continue
        day = rng.randint(1, 28)
        cursor = starts[0].replace(day=min(day, _days_in_month(starts[0])))
        while cursor < anchor:
            amt = _amount(rng, spec)
            # §7 price_hike: Spotify rises 15% for the final two months.
            if spec is hike_spec and cursor >= starts[-2]:
                amt = round(amt * 1.15, 2)
                manifest.price_hike = {
                    "merchant": spec.canonical,
                    "typical_amount": -spec.amount[0],
                    "hiked_amount": amt,
                    "from_period": starts[-2].strftime("%Y-%m"),
                }
            emit(spec, cursor, amt)
            cursor += timedelta(days=spec.cadence_days + rng.randint(-1, 1))

    # --- frequent / occasional spend ---
    spike_month = starts[-4]
    for month_start in starts:
        span = _days_in_month(month_start)
        for spec in CATALOG:
            if spec.kind == "recurring":
                continue
            lo, hi = spec.per_month
            count = rng.randint(lo, hi)
            # §7 baseline_anomaly: one travel-heavy month, well past z=2.
            if spec.category == "travel" and month_start == spike_month:
                count += 4
            for _ in range(count):
                emit(spec, month_start + timedelta(days=rng.randint(0, span - 1)))

    if manifest.price_hike:
        manifest.spend_spike = {"category": "travel", "period": spike_month.strftime("%Y-%m")}

    # --- §7 new_merchant: Peloton only in the final month ---
    last = starts[-1]
    peloton_day = last.replace(day=min(12, _days_in_month(last)))
    emit(NEW_MERCHANT, peloton_day)
    manifest.new_merchant = {"merchant": NEW_MERCHANT.canonical, "period": last.strftime("%Y-%m")}

    # --- §7 duplicate: same merchant, same amount, inside 48h ---
    candidates = [r for r in rows if r["true_merchant"] == "Amazon" and r["type"] == "purchase"]
    original = rng.choice(candidates)
    dup_date = date.fromisoformat(original["posted_date"]) + timedelta(days=1)
    dup = dict(original)
    dup["posted_date"] = dup_date.isoformat()
    # Same amount, different descriptor — a real double-charge looks like this.
    dup["raw_descriptor"] = _descriptor(rng, next(s for s in CATALOG if s.canonical == "Amazon"), dup_date)
    rows.append(dup)
    manifest.duplicate = {
        "merchant": "Amazon",
        "amount": original["amount"],
        "dates": [original["posted_date"], dup["posted_date"]],
    }

    rows.sort(key=lambda r: (r["posted_date"], r["true_merchant"]))
    return rows, manifest


def write(out_dir: Path = OUT_DIR, seed: int = SEED) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest = generate(seed=seed)

    txn_path = out_dir / "transactions.csv"
    with txn_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.__dict__, indent=2) + "\n")

    labels_path = write_label_set(rows)

    return {
        "rows": len(rows),
        "months": MONTHS,
        "merchants": len({r["true_merchant"] for r in rows}),
        "transactions_csv": str(txn_path),
        "manifest_json": str(manifest_path),
        "labeled_csv": str(labels_path),
    }


LABEL_SET_SIZE = 200
EVAL_DIR = Path(__file__).resolve().parent.parent / "evals"


def write_label_set(rows: list[dict], n: int = LABEL_SET_SIZE, out_dir: Path = EVAL_DIR) -> Path:
    """Emit §9.2's labeled set from generator ground truth.

    This is not circular, and the distinction matters if anyone asks. The labels
    come from the generator; the categorizer under test sees only
    `raw_descriptor` and must recover merchant and category from that string
    alone. It never reads true_merchant/true_category. So the measurement is a
    real one: can the tiered pipeline undo the noise the generator applied?

    What it does *not* prove is performance on real statements, whose noise this
    catalog only approximates. That still needs hand-labeling, and the README
    should say which number came from which.
    """
    rng = random.Random(SEED)
    purchases = [r for r in rows if r["type"] == "purchase"]
    sample = rng.sample(purchases, min(n, len(purchases)))
    sample.sort(key=lambda r: (r["true_merchant"], r["posted_date"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "labeled_categories.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["raw_descriptor", "expected_merchant", "expected_category"])
        w.writeheader()
        for r in sample:
            w.writerow({
                "raw_descriptor": r["raw_descriptor"],
                "expected_merchant": r["true_merchant"],
                "expected_category": r["true_category"],
            })
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic statements (§10)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    summary = write(out_dir=args.out, seed=args.seed)
    for k, v in summary.items():
        print(f"{k:18} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
