-- SAI Inference Detection Tracking Database Setup
-- Clean implementation for PostgreSQL production deployment

-- Create user and grant permissions
CREATE USER IF NOT EXISTS sai_user WITH PASSWORD 'SAI_2024_DetectionsDB!';
GRANT CONNECT ON DATABASE sai_dashboard TO sai_user;
GRANT CREATE ON SCHEMA public TO sai_user;
GRANT USAGE ON SCHEMA public TO sai_user;

-- Grant permissions on existing tables in case they exist
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO sai_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO sai_user;

-- Grant permissions on future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO sai_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO sai_user;

-- Create camera detections table
CREATE TABLE IF NOT EXISTS camera_detections (
    id SERIAL PRIMARY KEY,
    camera_id VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    detection_count INTEGER DEFAULT 1 CHECK (detection_count > 0),
    created_at TIMESTAMP DEFAULT NOW(),
    metadata TEXT
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_camera_detections_camera_id ON camera_detections(camera_id);
CREATE INDEX IF NOT EXISTS idx_camera_detections_created_at ON camera_detections(created_at);
CREATE INDEX IF NOT EXISTS idx_camera_detections_camera_time ON camera_detections(camera_id, created_at);

-- Add comments for documentation
COMMENT ON TABLE camera_detections IS 'Raw wildfire smoke detection facts for query-time alert calculation';
COMMENT ON COLUMN camera_detections.camera_id IS 'Unique identifier for camera location';
COMMENT ON COLUMN camera_detections.confidence IS 'Detection confidence score (0.0-1.0)';
COMMENT ON COLUMN camera_detections.detection_count IS 'Number of detections in this batch';
COMMENT ON COLUMN camera_detections.created_at IS 'Timestamp when detection occurred';
COMMENT ON COLUMN camera_detections.metadata IS 'Additional detection metadata (JSON string)';

-- Grant final permissions on the new table
GRANT SELECT, INSERT, UPDATE, DELETE ON camera_detections TO sai_user;
GRANT USAGE, SELECT ON SEQUENCE camera_detections_id_seq TO sai_user;

-- Display setup summary
SELECT
    'Database setup completed' AS status,
    current_database() AS database_name,
    current_user AS current_user,
    now() AS setup_time;

-- Show table structure
\d camera_detections;