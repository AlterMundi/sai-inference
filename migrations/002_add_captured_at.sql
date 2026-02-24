-- Migration 002: Add captured_at column to camera_detections
-- This column stores the actual capture time from the camera node,
-- as opposed to created_at which is the server arrival time.

ALTER TABLE camera_detections
  ADD COLUMN IF NOT EXISTS captured_at TIMESTAMPTZ;

-- Indexes use CONCURRENTLY and must run outside a transaction (autocommit).
-- The application code handles this specially.
-- CONCURRENT_INDEX: CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detections_captured_at ON camera_detections (captured_at);
-- CONCURRENT_INDEX: CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detections_camera_captured ON camera_detections (camera_id, captured_at);
