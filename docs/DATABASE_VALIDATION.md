# Database & Logging System Validation Guide

## System Overview

SAI Inference Service uses PostgreSQL for comprehensive execution logging and temporal alert tracking. This document provides validation procedures and expected metrics.

## Database Configuration

**Database**: `sai_dashboard` (shared with n8n services)
**Table**: `camera_detections` (owner: `sai_user`)
**Connection Pool**: asyncpg (2-10 connections)
**Application Name**: `sai-inference-detections`

### Connection String
```bash
postgresql://sai_user:password@localhost/sai_dashboard?sslmode=disable
```

## Quick Validation Checklist

### 1. Database Connection
```bash
# Check active connections
sudo -u postgres psql -d sai_dashboard -c "SELECT application_name, state FROM pg_stat_activity WHERE application_name LIKE '%sai%';"

# Expected output:
#     application_name     | state
# --------------------------+-------
#  sai-inference-detections | idle
```

### 2. Table Existence & Schema
```bash
# Verify table structure
sudo -u postgres psql -d sai_dashboard -c "\d camera_detections"

# Expected: 25 columns, 11 indexes
```

### 3. Recent Activity
```bash
# Check recent records
sudo -u postgres psql -d sai_dashboard -c "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM camera_detections WHERE created_at > NOW() - INTERVAL '24 hours';"

# Expected: ~5,000 records/day
```

### 4. Detection Distribution
```bash
# Check detection vs no-detection ratio
sudo -u postgres psql -d sai_dashboard -c "SELECT COUNT(*) FILTER (WHERE detection_count > 0) as with_detections, COUNT(*) FILTER (WHERE detection_count = 0) as no_detections, COUNT(*) as total FROM camera_detections WHERE created_at > NOW() - INTERVAL '24 hours';"

# Expected: ~0.05-0.1% detection rate (high selectivity)
```

### 5. Escalation Events
```bash
# View recent escalations
sudo -u postgres psql -d sai_dashboard -c "SELECT camera_id, created_at, base_alert_level, final_alert_level, escalation_reason FROM camera_detections WHERE escalation_reason IS NOT NULL ORDER BY created_at DESC LIMIT 10;"

# Expected: <0.1% escalation rate
```

## Production Metrics (Baseline)

Based on 4 days of production operation (2025-10-03 to 2025-10-07):

### Execution Volume
- **Total Records**: 17,541 executions
- **Daily Average**: ~5,000 executions/day
- **Hourly Rate**: ~200-210 executions/hour
- **Peak Period**: 05:00-10:00 (camera activity)

### Detection Statistics
- **Detection Rate**: 0.074% (13 detections in 17,541 executions)
- **No Detections**: 99.926% (17,528 records)
- **Implication**: High selectivity, very low false positive rate

### Alert Distribution
```
Alert Type         | Count  | Percentage
-------------------|--------|------------
none → none        | 17,528 | 99.93%
low → high         |      9 |  0.05%  (persistence escalation)
high → high        |      4 |  0.02%  (direct high confidence)
```

### Performance Metrics
- **Average Latency**: 70-90ms per inference
- **Processing Time**: 80ms average (from `processing_time_ms` column)
- **Memory Usage**: 1.8GB stable
- **CPU Usage**: <1% per request

### Camera Activity
- **Active Cameras**: 6 cameras registered
- **Primary Camera**: sai-cam-node-07:cam1 (most active)
- **Execution Pattern**: ~5min intervals per camera

## API Validation

### Analytics Endpoints
```bash
# List active cameras
curl -s http://localhost:8888/api/v1/cameras?hours=24 | jq .

# Camera statistics
curl -s http://localhost:8888/api/v1/cameras/sai-cam-node-07:cam1/stats?hours=24 | jq .

# Alert summary
curl -s http://localhost:8888/api/v1/alerts/summary?hours=24 | jq .

# Escalation statistics
curl -s http://localhost:8888/api/v1/alerts/escalation-stats?hours=24 | jq .

# Prometheus metrics
curl -s http://localhost:8888/metrics | grep sai_
```

### Expected API Responses
```json
// /api/v1/cameras
[
  {
    "camera_id": "sai-cam-node-07:cam1",
    "last_detection": "2025-10-07T13:24:11.223713",
    "detection_count_24h": 839,
    "last_alert_level": "none"
  }
]

// /api/v1/alerts/summary
{
  "total_alerts": 5058,
  "by_level": {
    "high": 9,
    "none": 5049
  },
  "escalation_rate": 0.178,
  "cameras_active": 1
}
```

## Database Schema Details

### Key Columns
```sql
-- Identification
id SERIAL PRIMARY KEY
camera_id VARCHAR(100) NOT NULL
request_id VARCHAR(50)
created_at TIMESTAMP DEFAULT (NOW() AT TIME ZONE 'UTC')

-- Detection Summary
detection_count INTEGER DEFAULT 0
smoke_count INTEGER DEFAULT 0
fire_count INTEGER DEFAULT 0
max_confidence FLOAT NOT NULL
avg_confidence FLOAT

-- Alert State Tracking
base_alert_level VARCHAR(20) NOT NULL        -- Initial: none/low/high
final_alert_level VARCHAR(20) NOT NULL       -- After analysis: none/low/high/critical
escalation_reason VARCHAR(50)                -- persistence_low | persistence_high | false_positive_pattern

-- Spatial Data
detections JSONB  -- Array of {class_id, class_name, confidence, bbox: {x1, y1, x2, y2}}

-- Performance Metrics
processing_time_ms FLOAT
model_inference_time_ms FLOAT
image_width INTEGER
image_height INTEGER

-- Model Configuration
model_version VARCHAR(50)
confidence_threshold FLOAT
iou_threshold FLOAT
detection_classes INTEGER[]

-- Request Context
source VARCHAR(100)                 -- 'n8n', 'api-direct', 'mosaic', 'cli'
n8n_workflow_id VARCHAR(100)
n8n_execution_id VARCHAR(100)
metadata JSONB
```

### Indexes
```sql
idx_camera_time          -- (camera_id, created_at DESC)
idx_created_at           -- (created_at DESC)
idx_base_alert           -- (base_alert_level)
idx_final_alert          -- (final_alert_level)
idx_escalated            -- Partial: WHERE base_alert_level != final_alert_level
idx_detections_jsonb     -- GIN index for JSONB queries
idx_has_detections       -- Partial: WHERE detection_count > 0
idx_high_confidence      -- Partial: WHERE max_confidence >= 0.7
idx_source               -- (source)
idx_request_id           -- (request_id)
```

### Integrity Constraints
```sql
-- Alert state consistency
CHECK (
    (base_alert_level = final_alert_level AND escalation_reason IS NULL)
    OR
    (base_alert_level != final_alert_level AND escalation_reason IS NOT NULL)
)

-- Detection count consistency
CHECK (smoke_count + fire_count <= detection_count)

-- Confidence ranges
CHECK (max_confidence >= 0.0 AND max_confidence <= 1.0)
CHECK (avg_confidence IS NULL OR (avg_confidence >= 0.0 AND avg_confidence <= 1.0))
```

## Troubleshooting

### No Database Connection
```bash
# Check PostgreSQL service
sudo systemctl status postgresql

# Check database exists
sudo -u postgres psql -l | grep sai_dashboard

# Check user permissions
sudo -u postgres psql -d sai_dashboard -c "\du sai_user"
```

### No Recent Records
```bash
# Check sai-inference service status
sudo systemctl status sai-inference

# Check service logs
sudo journalctl -u sai-inference -n 100

# Verify database URL in config
cat /etc/sai-inference/production.env | grep DATABASE
```

### High Detection Rate (>1%)
This may indicate:
- Model threshold too low (check `confidence_threshold`)
- Environmental changes (lighting, weather)
- Camera position changes
- Potential false positives requiring investigation

### No Escalations
This is NORMAL for typical operation. Escalations only occur when:
- 3+ low-confidence detections within 30 minutes (persistence_low)
- 3+ high-confidence detections within 3 hours (persistence_high)

## Maintenance

### Regular Checks (Weekly)
```bash
# Database size
sudo -u postgres psql -d sai_dashboard -c "SELECT pg_size_pretty(pg_database_size('sai_dashboard'));"

# Record count growth
sudo -u postgres psql -d sai_dashboard -c "SELECT COUNT(*), DATE_TRUNC('day', created_at) as day FROM camera_detections GROUP BY day ORDER BY day DESC LIMIT 7;"

# Index health
sudo -u postgres psql -d sai_dashboard -c "SELECT schemaname, tablename, indexname FROM pg_indexes WHERE tablename = 'camera_detections';"
```

### Archival (Optional)
For long-term retention, consider archiving records older than 90 days:
```sql
-- Archive old records (example)
CREATE TABLE camera_detections_archive AS
SELECT * FROM camera_detections
WHERE created_at < NOW() - INTERVAL '90 days';

DELETE FROM camera_detections
WHERE created_at < NOW() - INTERVAL '90 days';

VACUUM ANALYZE camera_detections;
```

## Migration History

### Initial Deployment (2025-10-03)
- Migration: `001_enhanced_schema.sql`
- Features: 25-column schema with 11 indexes
- Alert system: Dual-window temporal tracking
- Performance: Optimized for 5K executions/day

### Future Migrations
Store in `/opt/sai-inference/migrations/` with sequential numbering:
- `002_*.sql` - Next schema change
- Include rollback procedures
- Update this document with changes

## Support

For issues or questions:
1. Check service logs: `sudo journalctl -u sai-inference -f`
2. Verify database connection: Use validation queries above
3. Check API health: `curl http://localhost:8888/api/v1/health`
4. Review metrics: `curl http://localhost:8888/metrics`

---

**Last Updated**: 2025-10-07
**Schema Version**: 001_enhanced_schema.sql
**Production Status**: ✅ Validated (4 days operation)
