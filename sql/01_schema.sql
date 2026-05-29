-- Prodapt Capstone - Database Schema
-- Run against SQLite to create all operational tables.
-- Students must use these tables for ALL agent tools — do NOT use in-memory dictionaries.

-- ---------------------------------------------------------------------------
-- Network operations (LlamaIndex semantic SQL + Network Diagnostics ADK tools)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS network_towers (
    tower_id              TEXT PRIMARY KEY,
    region                TEXT NOT NULL,
    city                  TEXT NOT NULL,
    technology            TEXT NOT NULL,   -- LTE, 5G
    status                TEXT NOT NULL,   -- OPERATIONAL, DEGRADED, OUTAGE, MAINTENANCE
    capacity_subscribers  INTEGER,
    last_maintenance      DATE
);

CREATE TABLE IF NOT EXISTS network_outages (
    outage_id             TEXT PRIMARY KEY,
    region                TEXT NOT NULL,
    tower_id              TEXT,
    start_time            TEXT NOT NULL,
    end_time              TEXT,
    severity              TEXT NOT NULL,   -- CRITICAL, HIGH, MEDIUM, LOW
    affected_customers    INTEGER,
    root_cause            TEXT,
    incident_id           TEXT,
    FOREIGN KEY (tower_id) REFERENCES network_towers(tower_id)
);

CREATE TABLE IF NOT EXISTS tower_performance (
    record_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    tower_id              TEXT NOT NULL,
    avg_latency_ms        REAL,
    packet_loss_pct       REAL,
    throughput_mbps       REAL,
    signal_strength_dbm   REAL,
    recorded_at           TEXT NOT NULL,
    FOREIGN KEY (tower_id) REFERENCES network_towers(tower_id)
);

CREATE TABLE IF NOT EXISTS open_incidents (
    incident_id           TEXT PRIMARY KEY,
    tower_id              TEXT NOT NULL,
    severity              TEXT NOT NULL,
    status                TEXT NOT NULL DEFAULT 'OPEN',  -- OPEN, IN_PROGRESS, RESOLVED
    description           TEXT,
    eta_hours             INTEGER,
    opened_at             TEXT NOT NULL,
    FOREIGN KEY (tower_id) REFERENCES network_towers(tower_id)
);

-- ---------------------------------------------------------------------------
-- Customer subscriptions (LlamaIndex semantic SQL analytics)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS customer_subscriptions (
    customer_id           TEXT PRIMARY KEY,
    account_type          TEXT NOT NULL,   -- Consumer, Business, Enterprise
    plan_name             TEXT NOT NULL,
    data_limit_gb         REAL,
    monthly_fee           REAL,
    contract_end          DATE,
    region                TEXT
);

-- ---------------------------------------------------------------------------
-- Billing (Billing Resolution ADK tools — all reads/writes go to SQL)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS billing_accounts (
    customer_id           TEXT PRIMARY KEY,
    current_balance       REAL NOT NULL DEFAULT 0,
    billing_cycle_day     INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customer_subscriptions(customer_id)
);

CREATE TABLE IF NOT EXISTS billing_charges (
    charge_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id           TEXT NOT NULL,
    description           TEXT NOT NULL,
    amount                REAL NOT NULL,
    charge_date           TEXT NOT NULL,
    billing_period        TEXT,
    is_duplicate_flag     INTEGER DEFAULT 0,  -- 0 = valid, 1 = flagged duplicate
    FOREIGN KEY (customer_id) REFERENCES customer_subscriptions(customer_id)
);

CREATE TABLE IF NOT EXISTS billing_credits (
    credit_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id           TEXT NOT NULL,
    amount                REAL NOT NULL,
    reason                TEXT NOT NULL,
    status                TEXT NOT NULL DEFAULT 'APPLIED',  -- APPLIED, PENDING_APPROVAL
    reference_number      TEXT,
    applied_at            TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customer_subscriptions(customer_id)
);

CREATE TABLE IF NOT EXISTS billing_disputes (
    dispute_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id           TEXT NOT NULL,
    charge_id             INTEGER,
    dispute_reason        TEXT NOT NULL,
    status                TEXT NOT NULL DEFAULT 'OPEN',  -- OPEN, RESOLVED, REJECTED
    opened_at             TEXT NOT NULL,
    resolved_at           TEXT,
    FOREIGN KEY (customer_id) REFERENCES customer_subscriptions(customer_id),
    FOREIGN KEY (charge_id) REFERENCES billing_charges(charge_id)
);
