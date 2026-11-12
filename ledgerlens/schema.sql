-- LedgerLens schema — §4 of instructions.md
-- Conventions locked here: amounts negative for outflow, dates ISO strings,
-- every analytics query filters type = 'purchase'.

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
