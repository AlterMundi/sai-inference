-- Migration 003: Backfill captured_at from created_at for historical rows
-- Best-effort: no metadata available for old rows, so created_at is the best proxy.

UPDATE camera_detections
  SET captured_at = created_at
WHERE captured_at IS NULL;
