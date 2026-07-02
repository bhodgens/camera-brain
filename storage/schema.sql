-- Camera Brain Database Schema
-- PostgreSQL 16 + TimescaleDB + pgvector

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Cameras table
CREATE TABLE IF NOT EXISTS cameras (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name text NOT NULL UNIQUE,
    rtsp_url text NOT NULL,
    location text,
    active boolean DEFAULT true,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

-- Workers table
CREATE TABLE IF NOT EXISTS workers (
    id text PRIMARY KEY,
    last_heartbeat timestamptz NOT NULL,
    status text NOT NULL DEFAULT 'offline',
    current_camera_id uuid REFERENCES cameras(id),
    assigned_at timestamptz,
    created_at timestamptz DEFAULT now()
);

-- Observations table (main detection storage)
CREATE TABLE IF NOT EXISTS observations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id uuid REFERENCES cameras(id),
    detected_at timestamptz NOT NULL,
    ingested_at timestamptz DEFAULT now(),
    type text NOT NULL,
    bbox jsonb,
    confidence float,
    class_id integer,
    class_name text,
    description text,
    attributes jsonb,
    embedding vector(1024),
    crop_path text,
    crop_retained boolean DEFAULT false,
    person_id uuid,
    is_new_person boolean DEFAULT false
);

-- Convert observations to hypertable
SELECT create_hypertable('observations', 'detected_at', if_not_exists => TRUE);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_observations_camera ON observations(camera_id, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_observations_type ON observations(type, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_observations_person ON observations(person_id) WHERE person_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_observations_attributes ON observations USING GIN(attributes);
CREATE INDEX IF NOT EXISTS idx_observations_embedding ON observations USING ivfflat(embedding vector_cosine_ops) WHERE embedding IS NOT NULL;

-- Persons table (known identities)
CREATE TABLE IF NOT EXISTS persons (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name text NOT NULL UNIQUE,
    embedding_centroid vector(1024),
    created_at timestamptz DEFAULT now(),
    last_seen_at timestamptz,
    metadata jsonb DEFAULT '{}'::jsonb
);

-- Activity summaries from LLM
CREATE TABLE IF NOT EXISTS activity_summaries (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id uuid REFERENCES cameras(id),
    period_start timestamptz NOT NULL,
    period_end timestamptz NOT NULL,
    summary_text text,
    embedding vector(1024),
    UNIQUE(camera_id, period_start)
);

-- Continuous aggregate: detections per hour
CREATE MATERIALIZED VIEW IF NOT EXISTS detections_per_hour
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', detected_at) AS bucket,
    camera_id,
    type,
    class_name,
    count(*) AS detection_count
FROM observations
GROUP BY bucket, camera_id, type, class_name;

-- Continuous aggregate: person appearances per day
CREATE MATERIALIZED VIEW IF NOT EXISTS person_daily_count
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', detected_at) AS bucket,
    person_id,
    count(*) AS appearances,
    count(DISTINCT detected_at::date) AS distinct_days
FROM observations
WHERE person_id IS NOT NULL
GROUP BY bucket, person_id;
