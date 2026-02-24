-- Migration 004: Enforce NOT NULL on captured_at after backfill

ALTER TABLE camera_detections
  ALTER COLUMN captured_at SET NOT NULL;
