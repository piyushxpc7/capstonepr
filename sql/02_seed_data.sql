-- Prodapt Capstone - Seed Data
-- Run after 01_schema.sql against SQLite (data/telecom_ops.db recommended).

-- Clear existing data (order respects foreign keys)
DELETE FROM billing_disputes;
DELETE FROM billing_credits;
DELETE FROM billing_charges;
DELETE FROM billing_accounts;
DELETE FROM open_incidents;
DELETE FROM tower_performance;
DELETE FROM network_outages;
DELETE FROM customer_subscriptions;
DELETE FROM network_towers;

-- ---------------------------------------------------------------------------
-- Network towers
-- ---------------------------------------------------------------------------
INSERT INTO network_towers (tower_id, region, city, technology, status, capacity_subscribers, last_maintenance) VALUES
('TX-512', 'Southwest',  'Austin',       '5G',  'OPERATIONAL', 12000, '2026-03-15'),
('TX-518', 'Southwest',  'Austin',       'LTE', 'DEGRADED',     8000,  '2026-01-20'),
('TX-601', 'Southwest',  'Dallas',       '5G',  'OPERATIONAL', 15000, '2026-02-10'),
('MW-102', 'Midwest',    'Chicago',      '5G',  'OPERATIONAL', 18000, '2026-04-01'),
('MW-115', 'Midwest',    'Detroit',      'LTE', 'OUTAGE',       9000,  '2025-11-05'),
('NE-201', 'Northeast',  'Boston',       '5G',  'OPERATIONAL', 14000, '2026-03-28'),
('NE-210', 'Northeast',  'New York',     '5G',  'OPERATIONAL', 22000, '2026-02-22'),
('WE-301', 'West',       'Los Angeles',  '5G',  'OPERATIONAL', 25000, '2026-04-05'),
('WE-318', 'West',       'Seattle',      'LTE', 'MAINTENANCE',  11000, '2026-05-01'),
('SE-401', 'Southeast',  'Atlanta',      '5G',  'OPERATIONAL', 16000, '2026-03-12');

-- ---------------------------------------------------------------------------
-- Network outages (historical)
-- ---------------------------------------------------------------------------
INSERT INTO network_outages (outage_id, region, tower_id, start_time, end_time, severity, affected_customers, root_cause, incident_id) VALUES
('OUT-2026-0412', 'Midwest',   'MW-115', '2026-04-12T14:00:00', '2026-04-12T20:00:00', 'CRITICAL', 8500, 'Fiber backhaul cut - construction accident', 'INC0048123'),
('OUT-2026-0428', 'Southwest', 'TX-518', '2026-04-28T14:00:00', '2026-04-28T17:00:00', 'HIGH',     3200, 'Power outage - generator failure',           'INC0049201'),
('OUT-2026-0505', 'Northeast', 'NE-210', '2026-05-05T14:00:00', '2026-05-05T16:00:00', 'MEDIUM',   1100, 'RRU hardware failure',                     'INC0050102'),
('OUT-2026-0510', 'West',      'WE-301', '2026-05-10T14:00:00', '2026-05-10T15:00:00', 'LOW',       450, 'Software patch rollback',                  'INC0050444'),
('OUT-2026-0515', 'Midwest',   'MW-102', '2026-05-15T14:00:00', '2026-05-15T18:00:00', 'HIGH',     5600, 'Lightning strike - antenna damage',        'INC0050789'),
('OUT-2026-0518', 'Southeast', 'SE-401', '2026-05-18T14:00:00', NULL,                  'MEDIUM',    980, 'Under investigation',                      'INC0050999');

-- ---------------------------------------------------------------------------
-- Open incidents (for Network Diagnostics ADK agent — query/update via SQL)
-- ---------------------------------------------------------------------------
INSERT INTO open_incidents (incident_id, tower_id, severity, status, description, eta_hours, opened_at) VALUES
('INC0048123', 'MW-115', 'CRITICAL', 'IN_PROGRESS', 'Fiber backhaul cut - construction accident', 2,  '2026-04-12T14:05:00'),
('INC0049201', 'TX-518', 'HIGH',     'IN_PROGRESS', 'Power outage - generator failure',           6,  '2026-04-28T14:10:00'),
('INC0050999', 'SE-401', 'MEDIUM',   'OPEN',        'Intermittent 5G drops - under investigation', 4, '2026-05-18T14:00:00');

-- ---------------------------------------------------------------------------
-- Tower performance (latest metrics per tower)
-- ---------------------------------------------------------------------------
INSERT INTO tower_performance (tower_id, avg_latency_ms, packet_loss_pct, throughput_mbps, signal_strength_dbm, recorded_at) VALUES
('TX-512',  18.5,   0.12, 420.0, -72,  '2026-05-20T12:00:00'),
('TX-518',  85.2,   2.40,  45.0, -95,  '2026-05-20T12:00:00'),
('MW-115', 999.0, 100.00,   0.0, NULL, '2026-05-20T13:00:00'),
('MW-102',  22.1,   0.08, 380.0, -70,  '2026-05-20T12:00:00'),
('NE-210',  15.8,   0.05, 510.0, -69,  '2026-05-20T11:00:00'),
('WE-301',  20.3,   0.15, 395.0, -71,  '2026-05-20T12:00:00'),
('SE-401',  45.6,   1.20, 180.0, -82,  '2026-05-20T13:00:00');

-- ---------------------------------------------------------------------------
-- Customer subscriptions
-- ---------------------------------------------------------------------------
INSERT INTO customer_subscriptions (customer_id, account_type, plan_name, data_limit_gb, monthly_fee, contract_end, region) VALUES
('CUST-10001', 'Business',   'Business Pro 5G',    100.0, 89.99,  '2027-06-30', 'Northeast'),
('CUST-10002', 'Consumer',   'Unlimited Plus',      50.0, 65.99,  '2026-12-31', 'Southwest'),
('CUST-10003', 'Consumer',   'Standard Unlimited',  35.0, 55.99,  '2026-08-15', 'Midwest'),
('CUST-10004', 'Enterprise', 'Enterprise 5G',      500.0, 249.99, '2028-03-01', 'West'),
('CUST-10005', 'Consumer',   'Unlimited Plus',      50.0, 65.99,  '2027-01-20', 'Southeast'),
('CUST-10006', 'Business',   'Business Pro 5G',    100.0, 89.99,  '2027-09-30', 'Southwest'),
('CUST-10007', 'Consumer',   'Prepaid Basic',        5.0, 25.99,  '2026-06-30', 'Midwest'),
('CUST-10008', 'Consumer',   'Unlimited Plus',      50.0, 65.99,  '2027-04-15', 'Northeast');

-- ---------------------------------------------------------------------------
-- Billing accounts (Billing Resolution ADK agent — all reads/writes via SQL)
-- ---------------------------------------------------------------------------
INSERT INTO billing_accounts (customer_id, current_balance, billing_cycle_day) VALUES
('CUST-10001',  89.99, 15),
('CUST-10002', 131.98,  1),
('CUST-10003',  89.98, 10),
('CUST-10004', 249.99, 20),
('CUST-10005',  65.99,  5),
('CUST-10006',  89.99, 15),
('CUST-10007',  25.99,  1),
('CUST-10008',  65.99, 12);

-- ---------------------------------------------------------------------------
-- Billing charges (includes deliberate duplicate for CUST-10002 demo scenario)
-- ---------------------------------------------------------------------------
INSERT INTO billing_charges (customer_id, description, amount, charge_date, billing_period, is_duplicate_flag) VALUES
('CUST-10002', 'Unlimited Plus Plan',              65.99, '2026-05-01', '2026-05', 0),
('CUST-10002', 'Unlimited Plus Plan (duplicate)',  65.99, '2026-05-01', '2026-05', 1),
('CUST-10003', 'Standard Unlimited',               55.99, '2026-05-01', '2026-05', 0),
('CUST-10003', 'Roaming - Zone C Data 2.1GB',      31.50, '2026-05-03', '2026-05', 0),
('CUST-10003', 'Taxes and fees',                    2.49, '2026-05-01', '2026-05', 0),
('CUST-10001', 'Business Pro 5G',                  89.99, '2026-05-01', '2026-05', 0);
