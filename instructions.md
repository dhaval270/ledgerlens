# LedgerLens — Build Specification

Agentic personal expense analyst over your own bank statements. Local-first, verifiable, evaluated.

---

## 1. Scope

**Is:** an agent that ingests bank statements, normalizes them into SQLite, answers natural-language questions with verified numbers, and proactively surfaces anomalies each month.

**Is not:** connected to a bank API, capable of moving money, or a financial advisor. It reads statements you hand it and reports. Every state change requires your approval.

**Non-negotiables:**

1. The LLM never performs arithmetic. It writes SQL; SQLite computes.
2. Deterministic code first, LLM only on fallback. Log which path resolved each row.
3. Every number in an answer must trace to a tool result, or the answer is rejected.
4. Nothing leaves the machine. Local models only.

---

## 2. Stack

| Layer | Choice | Note |
|---|---|---|
| Orchestration | LangGraph | Not LangChain chains — you need cycles and interrupts |
| LLM | Ollama (`llama3.1:8b` or `qwen2.5:7b`) | Instruct-tuned; 7–8B is enough for SQL gen |
| Store | SQLite | Single file, zero setup, real SQL |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local, fast |
| Vector index | `sqlite-vec` or FAISS | Keep it in-process |
| PDF parsing | `pdfplumber` primary, `camelot` for ruled tables | |
| API | FastAPI | Consistent with your other project |
| Tracing | LangSmith or a local JSONL trace log | You must be able to show token cost per query |

Pin versions in `requirements.txt`. LangGraph's API moves.

---

## 3. Repo structure

```
ledgerlens/
├── README.md
├── requirements.txt
├── data/
│   ├── synthetic/          # committed — demo data
│   └── private/            # .gitignored — your real statements
├── ledgerlens/
│   ├── schema.sql
│   ├── db.py               # connection, read-only helper
│   ├── ingest/
│   │   ├── parse.py        # PDF/CSV → raw rows
│   │   ├── dedup.py        # content hashing
│   │   ├── merchants.py    # normalization tiers
│   │   ├── categorize.py   # categorization tiers
│   │   └── recurring.py    # periodicity detection
│   ├── agent/
│   │   ├── state.py        # AgentState TypedDict
│   │   ├── graph.py        # StateGraph wiring
│   │   └── nodes/
│   │       ├── planner.py
│   │       ├── sql_tool.py
│   │       ├── semantic_tool.py
│   │       ├── anomaly_tool.py
│   │       └── verifier.py
│   ├── proactive/
│   │   ├── checks.py       # deterministic detectors
│   │   └── digest.py       # LLM narration
│   └── api/main.py
├── evals/
│   ├── golden_queries.jsonl
│   ├── labeled_categories.csv
│   └── run_evals.py
└── tests/
```

`data/private/` in `.gitignore`. Never commit real statements.

---

## 4. Schema

```sql
PRAGMA foreign_keys = ON;

CREATE TABLE accounts (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    institution   TEXT,
    last4         TEXT,
    currency      TEXT NOT NULL DEFAULT 'USD'
);

CREATE TABLE categories (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    parent_id     INTEGER REFERENCES categories(id),
    is_essential  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE merchants (
    id                INTEGER PRIMARY KEY,
    canonical_name    TEXT NOT NULL UNIQUE,
    default_category_id INTEGER REFERENCES categories(id),
    first_seen        TEXT,
    last_seen         TEXT
);

-- every raw descriptor string ever mapped to a merchant
CREATE TABLE merchant_aliases (
    id            INTEGER PRIMARY KEY,
    raw_pattern   TEXT NOT NULL UNIQUE,
    merchant_id   INTEGER NOT NULL REFERENCES merchants(id),
    resolved_by   TEXT NOT NULL CHECK (resolved_by IN ('rule','llm','user')),
    confidence    REAL,
    created_at    TEXT NOT NULL
);

CREATE TABLE transactions (
    id             INTEGER PRIMARY KEY,
    account_id     INTEGER NOT NULL REFERENCES accounts(id),
    posted_date    TEXT NOT NULL,              -- ISO YYYY-MM-DD
    amount         REAL NOT NULL,              -- negative = outflow
    currency       TEXT NOT NULL DEFAULT 'USD',
    raw_descriptor TEXT NOT NULL,
    merchant_id    INTEGER REFERENCES merchants(id),
    category_id    INTEGER REFERENCES categories(id),
    raw_category   TEXT,                       -- pre-canonical label
    type           TEXT NOT NULL DEFAULT 'purchase'
                   CHECK (type IN ('purchase','transfer','income','fee','refund')),
    recurring_id   INTEGER REFERENCES recurring_series(id),
    source_file    TEXT NOT NULL,
    content_hash   TEXT NOT NULL UNIQUE,       -- idempotency
    categorized_by TEXT CHECK (categorized_by IN ('rule','learned','llm','user')),
    confidence     REAL,
    created_at     TEXT NOT NULL
);

CREATE INDEX idx_txn_date     ON transactions(posted_date);
CREATE INDEX idx_txn_merchant ON transactions(merchant_id);
CREATE INDEX idx_txn_category ON transactions(category_id);
CREATE INDEX idx_txn_type     ON transactions(type);

CREATE TABLE category_rules (
    id           INTEGER PRIMARY KEY,
    pattern      TEXT NOT NULL,               -- regex on raw_descriptor
    category_id  INTEGER NOT NULL REFERENCES categories(id),
    priority     INTEGER NOT NULL DEFAULT 100
);

-- user corrections: this is the memory that makes it "learn"
CREATE TABLE corrections (
    id             INTEGER PRIMARY KEY,
    merchant_id    INTEGER REFERENCES merchants(id),
    raw_descriptor TEXT,
    category_id    INTEGER NOT NULL REFERENCES categories(id),
    created_at     TEXT NOT NULL
);

CREATE TABLE recurring_series (
    id             INTEGER PRIMARY KEY,
    merchant_id    INTEGER NOT NULL REFERENCES merchants(id),
    cadence_days   INTEGER NOT NULL,
    typical_amount REAL NOT NULL,
    last_amount    REAL,
    last_seen      TEXT,
    active         INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE budgets (
    id           INTEGER PRIMARY KEY,
    category_id  INTEGER NOT NULL REFERENCES categories(id),
    period       TEXT NOT NULL DEFAULT 'monthly',
    limit_amount REAL NOT NULL,
    active_from  TEXT NOT NULL
);

CREATE TABLE insights (
    id           INTEGER PRIMARY KEY,
    kind         TEXT NOT NULL,               -- price_hike | duplicate | overspend | new_merchant
    subject_id   INTEGER,
    period       TEXT NOT NULL,               -- YYYY-MM
    payload      TEXT NOT NULL,               -- JSON
    surfaced_at  TEXT NOT NULL,
    dismissed    INTEGER NOT NULL DEFAULT 0,
    UNIQUE(kind, subject_id, period)          -- never repeat the same insight
);

CREATE TABLE query_log (
    id            INTEGER PRIMARY KEY,
    question      TEXT NOT NULL,
    plan          TEXT,                       -- JSON
    sql_attempts  INTEGER NOT NULL DEFAULT 0,
    verifier_pass INTEGER,
    latency_ms    INTEGER,
    tokens_in     INTEGER,
    tokens_out    INTEGER,
    created_at    TEXT NOT NULL
);
```

**Conventions to lock now:** amounts negative for outflow; dates ISO strings; every analytics query filters `type = 'purchase'`.

---

## 5. Ingestion pipeline

Deterministic. No agent. Runs on a watched folder or CLI command.

### 5.1 Parse
`pdfplumber` for text-layer PDFs, `camelot` for ruled tables, plain `csv` when available. Output raw rows: `(date, descriptor, amount)`.

Bank formats vary wildly. Write one adapter per institution under `ingest/parse.py` and dispatch on a detected header signature. Don't build a universal parser.

### 5.2 Dedup
```
content_hash = sha256(f"{account_id}|{posted_date}|{amount}|{raw_descriptor}")
```
Insert with `INSERT OR IGNORE`. Overlapping statement date ranges now cost nothing.

### 5.3 Merchant normalization (tiered)

1. **Alias lookup** — exact match on `merchant_aliases.raw_pattern`. Free.
2. **Rule strip** — remove known noise: `SQ *`, `TST*`, `PAYPAL *`, trailing store numbers, city/state tails, dates embedded in descriptors. Re-check alias table.
3. **LLM fallback** — only if 1 and 2 miss. Prompt: given raw descriptor, return canonical business name as JSON. Low temperature.
4. **Write back** — every LLM resolution is inserted into `merchant_aliases` with `resolved_by='llm'`. It never costs a call twice.

Track the tier hit rate. This becomes a resume number.

### 5.4 Categorization (tiered)

1. `corrections` table — you already fixed this merchant, use it. `categorized_by='user'`.
2. `merchants.default_category_id` — `'learned'`.
3. `category_rules` regex — `'rule'`.
4. LLM with the category list in the prompt — `'llm'`, store confidence.

Anything below your confidence threshold goes to a review queue rather than being silently guessed.

### 5.5 Recurring detection

Pure algorithm, no LLM. Group by `merchant_id`, sort dates, compute gaps. If ≥3 occurrences and the gap standard deviation is under ~20% of the mean, it's a series. Store cadence and typical amount. Flag when `last_amount` deviates from `typical_amount` by more than 5% — that's your subscription price-hike detector.

---

## 6. Agent graph

### 6.1 State

```python
class AgentState(TypedDict):
    question: str
    plan: list[dict]          # [{"sub_question", "tool", "rationale"}]
    tool_results: list[dict]  # [{"tool", "query", "rows", "error"}]
    draft_answer: str
    verifier_verdict: dict    # {"pass": bool, "reason": str}
    replan_count: int
    sql_attempts: int
```

### 6.2 Nodes and edges

```
question → planner
planner  → [sql_tool | semantic_tool | anomaly_tool]   (conditional, fan-out)
tools    → verifier
verifier → planner  (on fail, replan_count < 2)
verifier → answer   (on pass)
verifier → answer   (on fail + replan_count == 2, emitting an honest "couldn't verify")
```

### 6.3 Planner

Input: question + schema DDL + list of available tools + current date.
Output: strict JSON list of sub-questions, each tagged with a tool and a one-line rationale. No prose, no answer.

Routing guidance in the prompt:
- Anything countable, summable, or comparable → `sql`
- Free-text recall ("that trip to Boston", "anything medical-looking") → `semantic`
- "Is this unusual", "did anything change" → `anomaly`

Handle multi-part questions properly. "Did I spend more on food than last semester?" is two SQL calls plus a comparison, not one query.

### 6.4 SQL tool — self-correcting loop

```
generate SQL → execute on READ-ONLY connection
  ├─ success → return rows
  └─ exception → append error to prompt, regenerate (max 3 attempts)
       └─ exhausted → return {"error": "...", "rows": []}
```

Open the connection with `sqlite3.connect("file:ledger.db?mode=ro", uri=True)`. A malformed query then physically cannot mutate data.

Reject generated SQL containing `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ATTACH`, or `PRAGMA` before execution. Belt and braces.

Always return both the SQL and the rows in `tool_results` — the verifier needs both.

### 6.5 Semantic tool

Embed `raw_descriptor + merchant + category + month` per transaction. Retrieve top-k, return transaction IDs. Then hand those IDs to SQL for any aggregation — never let the embedding layer produce a number.

### 6.6 Anomaly tool

Per category, compute a rolling baseline from prior months. Flag via z-score or IQR. Returns structured findings, not prose. Reuse the anomaly-detection framing from your existing work.

### 6.7 Verifier

The differentiating node. Given `draft_answer` and `tool_results`:

1. Extract every numeric literal in the draft.
2. Check each against values present in `tool_results` rows (allow rounding tolerance).
3. Check no claim asserts something outside the retrieved date range.
4. Return `{"pass": bool, "reason": str}`.

Steps 1–3 are code, not LLM. Only the "reason" phrasing is generated. On failure, the reason feeds back into the planner as extra context.

If it fails twice, the answer says so plainly. An honest "I couldn't verify this" is a better demo than a confident wrong number — say that in the interview.

---

## 7. Proactive digest

Trigger: cron (monthly) or on successful ingestion.

Deterministic checks in `proactive/checks.py`:

| Check | Logic |
|---|---|
| Price hike | `recurring_series.last_amount` > `typical_amount` × 1.05 |
| Duplicate charge | same merchant + same amount within 48h |
| Category overspend | month-to-date sum > `budgets.limit_amount` |
| Baseline anomaly | category spend z-score > 2 vs trailing 6 months |
| New merchant | first `transactions` row for a merchant |
| Dormant subscription | series active but unused category for 90 days |

Each writes to `insights` with the `UNIQUE(kind, subject_id, period)` constraint preventing repeats. The LLM then narrates the undismissed insights into a short digest — it receives findings, never raw transactions.

---

## 8. Human-in-the-loop

Any proposed state change routes through LangGraph's `interrupt` and waits:

- Recategorize a merchant → writes to `corrections`
- Merge two merchant aliases
- Create or adjust a budget
- Dismiss an insight

Approvals are the only path to a write. Demo this explicitly — pause, show the pending diff, approve, show the effect.

---

## 9. Eval harness

Non-optional. This is what separates the project from a tutorial.

### 9.1 Golden query set
`evals/golden_queries.jsonl`, ~50 entries:
```json
{"question": "How much did I spend on groceries in March?",
 "expected_sql_shape": "SUM amount WHERE category=groceries AND month=03",
 "expected_value": -412.55,
 "tools": ["sql"]}
```
Cover: simple aggregates, date ranges, comparisons, multi-hop, semantic recall, and deliberately unanswerable questions (the correct answer is a refusal).

### 9.2 Categorization accuracy
Hand-label 200 transactions. Report accuracy overall and per resolution tier.

### 9.3 Metrics to report in the README

- Execution accuracy on golden set (% correct final value)
- SQL validity rate on first attempt vs. after repair
- Verifier catch rate — inject wrong numbers, confirm rejection
- Categorization accuracy, and % resolved without an LLM call
- Median latency and token cost per query

A README table of these numbers is worth more than another feature.

---

## 10. Privacy and demo

- Ollama only. State in the README that no financial data leaves the machine.
- Ship `data/synthetic/` — generate ~800 plausible transactions across 12 months with a script. Every screenshot, test, and demo uses synthetic data.
- `data/private/` gitignored, plus a pre-commit hook rejecting any file matching statement patterns.

---

## 11. Build order

| Phase | Deliverable | Gate |
|---|---|---|
| Week 1 | Schema, parser, dedup, merchant normalization, categorization | 200 labeled txns, accuracy measured |
| Week 2 | SQL tool + repair loop, planner, 50 golden queries | Execution accuracy reported |
| Week 3 | Verifier, semantic tool, FastAPI endpoint | Verifier catch rate reported |
| Week 4 | Proactive checks, digest, HITL approvals | End-to-end demo recorded |

Ship after Week 2 if time runs out. A working verified SQL agent beats a half-built five-node graph.

---

## 12. README must contain

1. One-paragraph problem statement
2. Architecture diagram (the agent graph)
3. The metrics table from §9.3
4. A 60-second demo GIF: question → plan → SQL → verified answer
5. An explicit "what this doesn't do" section — no bank API, no money movement, advisory only
6. Setup in under five commands

Point 5 reads as maturity, not weakness.

---

## 13. Resume bullets (fill the numbers after evals)

> **LedgerLens** — Local-first agentic expense analyst · Python, LangGraph, Ollama, SQLite, FastAPI
>
> - Built a multi-node LangGraph agent that decomposes natural-language finance questions, routes across SQL / semantic / statistical tools, and self-corrects failed queries — achieving __% execution accuracy on a 50-query held-out benchmark.
> - Engineered a verification node enforcing that every figure in a generated answer traces to a retrieved query result, cutting unverified numeric claims to __%.
> - Designed a tiered merchant-resolution and categorization pipeline (deterministic rules → learned mappings → LLM fallback) reaching __% accuracy on 200 hand-labeled transactions while resolving __% without an LLM call.
> - Implemented proactive monthly anomaly detection (subscription price changes, duplicate charges, baseline deviation) with human-in-the-loop approval gating all state changes.

Four bullets, all quantified, all defensible under questioning. Do not write a bullet you can't be interrogated on.

---

## 14. Interview talking points

- **"Why doesn't the LLM compute the totals?"** — Determinism and auditability. The model writes SQL; SQLite computes. A hallucinated number in a financial system is unacceptable, and this makes it structurally impossible.
- **"How do you know it's right?"** — Verifier node plus a 50-query benchmark, with numbers.
- **"What was hardest?"** — Merchant normalization. Bank descriptors are adversarially messy, and a naive LLM-per-row approach is both slow and non-reproducible. The tiered cache solved cost and consistency together.
- **"How would you scale it?"** — Postgres instead of SQLite, hosted model behind a gateway, async ingestion workers, per-user row-level isolation.
- **"What would you do differently?"** — Have an honest answer ready. Suggested: build the eval harness in week 1, not week 2.