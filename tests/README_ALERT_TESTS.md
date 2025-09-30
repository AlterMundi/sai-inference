# Enhanced Alert Management Test Suite

Comprehensive test suite for the SAI Inference Enhanced Alert Management System, covering alert escalation/de-escalation dynamics, temporal tracking, camera ID isolation, and database consistency.

## Overview

This test suite validates the complete alert management lifecycle:

- **Basic Alert Logic**: Confidence-based alert levels without temporal tracking
- **Enhanced Alert Logic**: Temporal tracking with camera ID persistence
- **Database Operations**: Storage, retrieval, and consistency of detection records
- **Escalation Dynamics**: Alert level progression based on persistence patterns
- **De-escalation**: Manual and automatic alert level reduction
- **Camera Isolation**: Multi-camera detection tracking without interference
- **System Statistics**: Aggregate metrics and monitoring

## Test Coverage

### 1. Basic Alert Level Tests (3 tests)

Tests fundamental alert logic without camera_id (basic mode):

- **test_basic_mode_no_camera_id**: Validates confidence-only alert levels
  - Low confidence (<0.3) → "none"
  - Medium confidence (0.3-0.7) → "low"
  - High confidence (≥0.7) → "high"

- **test_no_detections**: Ensures empty detection list returns "none"

- **test_fire_detections_ignored**: Verifies smoke-only wildfire mode
  - Fire detections ignored
  - Mixed fire/smoke uses only smoke confidence

### 2. Database Population and Consistency Tests (3 tests)

Validates database operations and data integrity:

- **test_database_population**: Verifies detection records correctly stored
  - Record structure validation
  - Field constraints (confidence 0-1, etc.)
  - Camera ID association

- **test_database_temporal_ordering**: Ensures correct time-based ordering
  - Descending timestamp order (most recent first)
  - Confidence sequence preservation

- **test_multiple_cameras_isolation**: Tests camera-specific data isolation
  - No cross-contamination between cameras
  - Independent detection counts per camera

### 3. Alert Escalation Tests (3 tests)

Tests progressive alert level increases based on persistence:

- **test_high_confidence_escalation**: High → Critical escalation
  - 1st high detection (≥0.7) → "high"
  - 2nd high detection → "high"
  - 3rd high detection → "critical" (3-hour persistence)

- **test_medium_confidence_escalation**: Low → High escalation
  - 1st medium detection (0.3-0.7) → "low"
  - 2nd medium detection → "low"
  - 3rd medium detection → "high" (30-minute persistence)

- **test_escalation_persistence_count**: Validates exact threshold behavior
  - Escalation requires `persistence_count` (default: 3) detections
  - No premature escalation

### 4. Alert De-escalation Tests (2 tests)

Tests alert level reduction mechanisms:

- **test_manual_deescalation_via_clear**: Manual intervention
  - Build to critical state
  - Clear camera history
  - Verify fresh start behavior

- **test_temporal_window_deescalation**: Automatic time-based de-escalation
  - Old detections (>3h for high, >30m for medium) ignored
  - New detections don't inherit old escalation state

### 5. Camera ID Temporal Tracking Tests (3 tests)

Validates per-camera detection tracking:

- **test_camera_status_query**: Status endpoint validation
  - Complete status structure
  - Confidence breakdown (high/medium/low counts)
  - Accurate recent detection counts

- **test_camera_first_detection**: Initial detection handling
  - Correct first detection storage
  - Proper alert level calculation

- **test_camera_no_history**: Empty state handling
  - Returns "none" for cameras with no detections
  - Null handling for missing data

### 6. Alert Lifecycle Tests (3 tests)

Tests complete detection-to-resolution workflows:

- **test_complete_alert_lifecycle**: Full lifecycle simulation
  1. Initial medium detection → "low"
  2. Persistence → "high"
  3. High-confidence upgrade → "high"
  4. High persistence → "critical"
  5. Manual clear → reset
  6. Fresh detection → "low"

- **test_intermittent_detection_pattern**: Non-continuous detections
  - Detections with gaps still escalate within time window
  - Real-world pattern validation

- **test_confidence_threshold_boundaries**: Exact threshold testing
  - Tests confidence values at/near 0.3 and 0.7 thresholds
  - Validates boundary condition handling

### 7. System Statistics Tests (1 test)

Validates system-wide monitoring:

- **test_system_statistics**: Aggregate metrics
  - Multi-camera activity tracking
  - Confidence distribution analysis
  - Active camera counts
  - Per-camera alert levels

### 8. Edge Cases and Error Handling (3 tests)

Tests resilience and error scenarios:

- **test_concurrent_detections_same_camera**: Race condition testing
  - 10 concurrent detections for same camera
  - No data loss or corruption
  - Proper record counts

- **test_database_unavailable_fallback**: Graceful degradation
  - Falls back to basic mode when DB unavailable
  - Still provides correct alert levels
  - No crashes or errors

- **test_multiple_detections_same_image**: Multi-detection handling
  - Uses maximum confidence from multiple detections
  - Correct detection_count tracking
  - Single database record per inference

## Alert Escalation Logic

### Thresholds (Configurable via `.env`)

```python
SAI_WILDFIRE_LOW_THRESHOLD=0.3   # Minimum confidence for tracking
SAI_WILDFIRE_HIGH_THRESHOLD=0.7  # High-confidence immediate alert
SAI_PERSISTENCE_COUNT=3          # Detections needed for escalation
SAI_ESCALATION_MINUTES=30        # Medium confidence time window
SAI_ESCALATION_HOURS=3           # High confidence time window
```

### Escalation Paths

#### Path 1: High Confidence (≥0.7)
```
Detection 1 → "high"
Detection 2 (within 3h) → "high"
Detection 3 (within 3h) → "critical"
```

#### Path 2: Medium Confidence (0.3-0.7)
```
Detection 1 → "low"
Detection 2 (within 30m) → "low"
Detection 3 (within 30m) → "high"
```

#### Path 3: Low Confidence (<0.3)
```
Any detection → "none" (not tracked)
```

## Database Schema

### Table: `camera_detections`

```sql
CREATE TABLE camera_detections (
    id SERIAL PRIMARY KEY,
    camera_id VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    detection_count INTEGER DEFAULT 1 CHECK (detection_count > 0),
    created_at TIMESTAMP DEFAULT NOW(),
    metadata TEXT,

    INDEX idx_camera_detections_camera_id (camera_id),
    INDEX idx_camera_detections_created_at (created_at),
    INDEX idx_camera_detections_camera_time (camera_id, created_at)
);
```

### Query Patterns

**Count recent high-confidence detections:**
```sql
SELECT COUNT(*)
FROM camera_detections
WHERE camera_id = 'cam_01'
  AND confidence >= 0.7
  AND created_at >= NOW() - INTERVAL '3 hours';
```

**Get recent detections:**
```sql
SELECT * FROM camera_detections
WHERE camera_id = 'cam_01'
  AND created_at >= NOW() - INTERVAL '30 minutes'
ORDER BY created_at DESC;
```

## Running the Tests

### Prerequisites

```bash
# Install dependencies
pip install pytest pytest-asyncio asyncpg

# Ensure PostgreSQL is running
systemctl status postgresql

# Create test database
sudo -u postgres psql -c "CREATE DATABASE sai_dashboard;"
sudo -u postgres psql -c "CREATE USER sai_user WITH PASSWORD 'password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE sai_dashboard TO sai_user;"
```

### Configuration

Create `.env` file with test database connection:

```bash
# Database (required for enhanced alert tests)
SAI_DATABASE_URL=postgresql://sai_user:password@localhost/sai_dashboard

# Alert thresholds
SAI_WILDFIRE_LOW_THRESHOLD=0.3
SAI_WILDFIRE_HIGH_THRESHOLD=0.7
SAI_PERSISTENCE_COUNT=3
SAI_ESCALATION_MINUTES=30
SAI_ESCALATION_HOURS=3
```

### Run All Tests

```bash
# Using pytest (recommended)
pytest tests/test_alert_management.py -v

# With coverage report
pytest tests/test_alert_management.py --cov=src.alert_manager --cov-report=html

# Run specific test
pytest tests/test_alert_management.py::test_high_confidence_escalation -v

# Run test group using markers
pytest tests/test_alert_management.py -k "escalation" -v

# Show print output (useful for debugging)
pytest tests/test_alert_management.py -v -s
```

### Alternative: Direct Python Execution

```bash
# If pytest not available, uses built-in async runner
python tests/test_alert_management.py
```

## Test Output Examples

### Successful Test Run

```
========================= test session starts =========================
tests/test_alert_management.py::test_basic_mode_no_camera_id PASSED
✅ Basic mode (no camera_id) works correctly

tests/test_alert_management.py::test_high_confidence_escalation PASSED
✅ High confidence escalation: high -> high -> critical

tests/test_alert_management.py::test_complete_alert_lifecycle PASSED
✅ Complete alert lifecycle: low -> high -> critical -> clear -> low

========================= 25 passed in 5.32s ==========================
🎉 ALL TESTS PASSED!
```

### Test Failure Example

```
tests/test_alert_management.py::test_escalation_persistence_count FAILED

AssertionError: Detection 3 should escalate to 'critical', got 'high'
Expected: critical
Actual: high

Potential causes:
- persistence_count threshold not reached
- Time window expired between detections
- Database query returned fewer records than expected
```

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Database State

```bash
# Connect to database
PGPASSWORD=password psql -U sai_user -d sai_dashboard

# Check detection records
SELECT camera_id, confidence, created_at
FROM camera_detections
ORDER BY created_at DESC
LIMIT 20;

# Count detections per camera
SELECT camera_id, COUNT(*) as count
FROM camera_detections
GROUP BY camera_id;

# Clean database between manual tests
DELETE FROM camera_detections;
```

### Common Issues

**Issue**: Tests fail with "database not initialized"
```bash
# Solution: Ensure PostgreSQL is running and .env is configured
systemctl start postgresql
echo "SAI_DATABASE_URL=postgresql://sai_user:password@localhost/sai_dashboard" >> .env
```

**Issue**: Race conditions in concurrent tests
```bash
# Solution: Increase asyncio delays if needed
await asyncio.sleep(0.1)  # Increase to 0.2 or 0.5
```

**Issue**: Temporal window tests fail inconsistently
```bash
# Solution: Tests manipulate timestamps directly to avoid timing issues
# If issues persist, check system clock synchronization
```

## Test Maintenance

### Adding New Tests

1. Follow existing pattern: `async def test_name(clean_database)`
2. Use `clean_database` fixture to ensure clean state
3. Use helper functions: `create_detection()`, `simulate_detection_sequence()`
4. Include descriptive assertions with error messages
5. Add print statements for successful validation

### Modifying Thresholds

If changing alert thresholds in `src/config.py`:

1. Update `.env` configuration
2. Review affected tests (search for hardcoded 0.3, 0.7 values)
3. Update test expectations accordingly
4. Re-run full test suite

### Performance Considerations

- Each test includes `asyncio.sleep()` to ensure temporal ordering
- Tests use minimal delays (0.05-0.2s) for speed
- Database cleaned between tests (each test is isolated)
- Full suite runs in ~5-10 seconds

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Alert Management Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: sai_dashboard
          POSTGRES_USER: sai_user
          POSTGRES_PASSWORD: password
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov

      - name: Run alert management tests
        env:
          SAI_DATABASE_URL: postgresql://sai_user:password@localhost/sai_dashboard
        run: |
          pytest tests/test_alert_management.py -v --cov=src.alert_manager

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Related Documentation

- [Alert Manager Implementation](../src/alert_manager.py)
- [Database Manager](../src/database.py)
- [Configuration Settings](../src/config.py)
- [Production Deployment](../deployment/README.md)

## Test Statistics

- **Total Tests**: 25
- **Test Categories**: 8
- **Lines of Test Code**: ~1000
- **Expected Duration**: 5-10 seconds
- **Coverage Target**: >90% of alert_manager.py

## Future Enhancements

Potential test additions:

1. **Load Testing**: Test with 100+ cameras and 10,000+ detections
2. **Stress Testing**: Concurrent detections across multiple cameras
3. **Time-Travel Testing**: Simulate days/weeks of detection patterns
4. **Failure Recovery**: Database connection loss during operations
5. **Performance Benchmarks**: Alert calculation latency metrics
6. **Integration Tests**: Full API endpoint testing with alert system

## License

This test suite is part of the SAI Inference Service project.
