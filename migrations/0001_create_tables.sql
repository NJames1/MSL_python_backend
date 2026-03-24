-- 0001_create_tables.sql
-- Create tables for MSL project

CREATE TABLE IF NOT EXISTS devices (
    id SERIAL PRIMARY KEY,
    device_hash VARCHAR(64) UNIQUE NOT NULL,
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE TABLE IF NOT EXISTS locations (
    id SERIAL PRIMARY KEY,
    centroid_lat NUMERIC(10,6),
    centroid_lon NUMERIC(10,6),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fingerprints (
    id SERIAL PRIMARY KEY,
    location_id INTEGER NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
    features JSONB,
    confidence NUMERIC(5,4),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE TABLE IF NOT EXISTS raw_scans (
    id SERIAL PRIMARY KEY,
    device_id INTEGER REFERENCES devices(id) ON DELETE SET NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
    cell_data JSONB,
    wifi_data JSONB,
    gps_lat NUMERIC(10,6),
    gps_lon NUMERIC(10,6)
);
