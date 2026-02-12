-- Run this in psql as a superuser (or a user with CREATE DATABASE privilege):
--   psql -h localhost -U postgres -f data/MEDIDOR/postgres/create_db_and_table.sql

-- 1) Create database for MEDIDOR data.
CREATE DATABASE medidor;

-- 2) Connect to the new database.
\connect medidor;

-- 3) Create table for internet speed samples.
CREATE TABLE IF NOT EXISTS internet_speed_samples (
    id BIGSERIAL PRIMARY KEY,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    download_mbps DOUBLE PRECISION NOT NULL,
    upload_mbps DOUBLE PRECISION NOT NULL,
    ping_ms DOUBLE PRECISION NOT NULL,
    server_name TEXT NOT NULL,
    server_country TEXT NOT NULL,
    isp TEXT NOT NULL,
    public_ip INET NOT NULL,
    local_hostname TEXT NOT NULL,
    machine TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Optional deduplication key. Prevents inserting the same sample twice.
CREATE UNIQUE INDEX IF NOT EXISTS ux_internet_speed_samples_natural
ON internet_speed_samples (timestamp_utc, public_ip, local_hostname);

-- 4) Create table for predicted samples (same schema as internet_speed_samples).
CREATE TABLE IF NOT EXISTS internet_speed_predictions (
    id BIGSERIAL PRIMARY KEY,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    download_mbps DOUBLE PRECISION NOT NULL,
    upload_mbps DOUBLE PRECISION NOT NULL,
    ping_ms DOUBLE PRECISION NOT NULL,
    server_name TEXT NOT NULL,
    server_country TEXT NOT NULL,
    isp TEXT NOT NULL,
    public_ip INET NOT NULL,
    local_hostname TEXT NOT NULL,
    machine TEXT NOT NULL,
    predicted_from_timestamp_utc TIMESTAMPTZ,
    target_metric TEXT,
    horizon_steps INTEGER,
    expected_step_seconds INTEGER,
    model_version TEXT,
    input_source TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_internet_speed_predictions_natural
ON internet_speed_predictions (timestamp_utc, public_ip, local_hostname);
