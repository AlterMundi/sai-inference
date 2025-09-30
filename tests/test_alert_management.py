#!/usr/bin/env python3
"""
Comprehensive test suite for Enhanced Alert Management System
Tests alert escalation/de-escalation dynamics, temporal tracking, and database consistency
"""
import asyncio
import pytest
import pytest_asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alert_manager import alert_manager
from src.database import db_manager
from src.models import Detection, BoundingBox, DetectionClass
from src.config import settings


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

@pytest_asyncio.fixture(scope="function")
async def clean_database():
    """Initialize and clean database before each test"""
    # Initialize if not already done
    if not db_manager.pool:
        await db_manager.initialize()

    # Clean all detection records
    async with db_manager.get_connection() as conn:
        await conn.execute("DELETE FROM camera_detections")

    yield

    # Cleanup after test (but don't close pool)
    try:
        async with db_manager.get_connection() as conn:
            await conn.execute("DELETE FROM camera_detections")
    except Exception:
        pass  # Ignore cleanup errors


def create_detection(
    class_name: str = "smoke",
    confidence: float = 0.5,
    bbox: tuple = (100, 100, 200, 200)
) -> Detection:
    """Create a test detection object"""
    return Detection(
        class_name=DetectionClass.SMOKE if class_name == "smoke" else DetectionClass.FIRE,
        class_id=0 if class_name == "smoke" else 1,
        confidence=confidence,
        bbox=BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
    )


async def simulate_detection_sequence(
    camera_id: str,
    confidences: List[float],
    delay_seconds: float = 0.1
) -> List[str]:
    """
    Simulate a sequence of detections with time delays
    Returns list of alert levels generated
    """
    alert_levels = []

    for confidence in confidences:
        detection = create_detection(confidence=confidence)
        alert_level = await alert_manager.determine_alert_level(
            detections=[detection],
            camera_id=camera_id
        )
        alert_levels.append(alert_level)

        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

    return alert_levels


# ============================================================================
# Basic Alert Level Tests
# ============================================================================

@pytest.mark.asyncio
async def test_basic_mode_no_camera_id(clean_database):
    """Test basic alert mode without camera_id (confidence-only)"""
    # Low confidence - should return "none"
    detections_low = [create_detection(confidence=0.2)]
    level = await alert_manager.determine_alert_level(detections_low, camera_id=None)
    assert level == "none", f"Expected 'none' for low confidence, got '{level}'"

    # Medium confidence - should return "low"
    detections_medium = [create_detection(confidence=0.4)]
    level = await alert_manager.determine_alert_level(detections_medium, camera_id=None)
    assert level == "low", f"Expected 'low' for medium confidence, got '{level}'"

    # High confidence - should return "high"
    detections_high = [create_detection(confidence=0.8)]
    level = await alert_manager.determine_alert_level(detections_high, camera_id=None)
    assert level == "high", f"Expected 'high' for high confidence, got '{level}'"

    print("✅ Basic mode (no camera_id) works correctly")


@pytest.mark.asyncio
async def test_no_detections(clean_database):
    """Test alert level with no detections"""
    level = await alert_manager.determine_alert_level([], camera_id="cam1")
    assert level == "none", f"Expected 'none' for no detections, got '{level}'"

    print("✅ No detections returns 'none'")


@pytest.mark.asyncio
async def test_fire_detections_ignored(clean_database):
    """Test that fire detections are ignored (smoke-only mode)"""
    # Fire detection only - should be ignored
    fire_detection = create_detection(class_name="fire", confidence=0.9)
    level = await alert_manager.determine_alert_level([fire_detection], camera_id="cam1")
    assert level == "none", f"Expected 'none' for fire-only, got '{level}'"

    # Mixed fire and smoke - only smoke should count
    smoke_detection = create_detection(class_name="smoke", confidence=0.5)
    mixed_detections = [fire_detection, smoke_detection]
    level = await alert_manager.determine_alert_level(mixed_detections, camera_id="cam2")
    assert level == "low", f"Expected 'low' for mixed detections, got '{level}'"

    print("✅ Fire detections correctly ignored in smoke-only mode")


# ============================================================================
# Database Population and Consistency Tests
# ============================================================================

@pytest.mark.asyncio
async def test_database_population(clean_database):
    """Test that detections are correctly stored in database"""
    camera_id = "test_cam_001"

    # Store multiple detections
    detection = create_detection(confidence=0.6)

    for i in range(5):
        await alert_manager.determine_alert_level([detection], camera_id=camera_id)
        await asyncio.sleep(0.05)

    # Verify database records
    records = await db_manager.get_detections_since(camera_id, minutes=10)

    assert len(records) == 5, f"Expected 5 records, got {len(records)}"

    # Check record structure
    for record in records:
        assert "camera_id" in record
        assert "confidence" in record
        assert "detection_count" in record
        assert "created_at" in record
        assert record["camera_id"] == camera_id
        assert 0 <= record["confidence"] <= 1

    print(f"✅ Database correctly stores {len(records)} detection records")


@pytest.mark.asyncio
async def test_database_temporal_ordering(clean_database):
    """Test that database maintains correct temporal ordering"""
    camera_id = "test_cam_temporal"
    confidences = [0.4, 0.5, 0.6, 0.7, 0.8]

    # Store detections with delays
    for conf in confidences:
        detection = create_detection(confidence=conf)
        await alert_manager.determine_alert_level([detection], camera_id=camera_id)
        await asyncio.sleep(0.1)

    # Retrieve records
    records = await db_manager.get_detections_since(camera_id, minutes=10)

    # Verify ordering (should be DESC by created_at)
    assert len(records) == 5

    # Check confidences are in reverse order (most recent first)
    stored_confidences = [r["confidence"] for r in records]
    expected_confidences = list(reversed(confidences))

    assert stored_confidences == expected_confidences, \
        f"Expected {expected_confidences}, got {stored_confidences}"

    # Verify timestamps are descending
    timestamps = [r["created_at"] for r in records]
    for i in range(len(timestamps) - 1):
        assert timestamps[i] >= timestamps[i+1], "Timestamps not in descending order"

    print("✅ Database maintains correct temporal ordering")


@pytest.mark.asyncio
async def test_multiple_cameras_isolation(clean_database):
    """Test that detections from different cameras are properly isolated"""
    cameras = ["cam_A", "cam_B", "cam_C"]

    # Store different number of detections per camera
    for idx, cam_id in enumerate(cameras):
        num_detections = (idx + 1) * 2  # 2, 4, 6 detections

        for i in range(num_detections):
            detection = create_detection(confidence=0.5)
            await alert_manager.determine_alert_level([detection], camera_id=cam_id)
            await asyncio.sleep(0.05)

    # Verify isolation
    for idx, cam_id in enumerate(cameras):
        expected_count = (idx + 1) * 2
        records = await db_manager.get_detections_since(cam_id, minutes=10)

        assert len(records) == expected_count, \
            f"Camera {cam_id}: expected {expected_count} records, got {len(records)}"

        # Verify all records belong to this camera
        for record in records:
            assert record["camera_id"] == cam_id, \
                f"Found record from wrong camera: {record['camera_id']}"

    print(f"✅ {len(cameras)} cameras correctly isolated")


# ============================================================================
# Alert Escalation Tests
# ============================================================================

@pytest.mark.asyncio
async def test_high_confidence_escalation(clean_database):
    """Test escalation from high to critical with persistence"""
    camera_id = "cam_escalation_high"
    high_threshold = settings.wildfire_high_threshold  # 0.7

    # First high-confidence detection -> should be "high"
    detection = create_detection(confidence=high_threshold)
    level1 = await alert_manager.determine_alert_level([detection], camera_id=camera_id)
    assert level1 == "high", f"First high detection should be 'high', got '{level1}'"

    await asyncio.sleep(0.1)

    # Second high-confidence detection -> still "high"
    level2 = await alert_manager.determine_alert_level([detection], camera_id=camera_id)
    assert level2 == "high", f"Second high detection should be 'high', got '{level2}'"

    await asyncio.sleep(0.1)

    # Third high-confidence detection -> should escalate to "critical"
    level3 = await alert_manager.determine_alert_level([detection], camera_id=camera_id)
    assert level3 == "critical", \
        f"Third high detection should escalate to 'critical', got '{level3}'"

    print(f"✅ High confidence escalation: high -> high -> critical")


@pytest.mark.asyncio
async def test_medium_confidence_escalation(clean_database):
    """Test escalation from medium confidence detections"""
    camera_id = "cam_escalation_medium"
    medium_confidence = 0.5  # Between low (0.3) and high (0.7)

    # First medium detection -> "low"
    detection = create_detection(confidence=medium_confidence)
    level1 = await alert_manager.determine_alert_level([detection], camera_id=camera_id)
    assert level1 == "low", f"First medium detection should be 'low', got '{level1}'"

    await asyncio.sleep(0.1)

    # Second medium detection -> still "low"
    level2 = await alert_manager.determine_alert_level([detection], camera_id=camera_id)
    assert level2 == "low", f"Second medium detection should be 'low', got '{level2}'"

    await asyncio.sleep(0.1)

    # Third medium detection -> should escalate to "high"
    level3 = await alert_manager.determine_alert_level([detection], camera_id=camera_id)
    assert level3 == "high", \
        f"Third medium detection should escalate to 'high', got '{level3}'"

    print(f"✅ Medium confidence escalation: low -> low -> high")


@pytest.mark.asyncio
async def test_escalation_persistence_count(clean_database):
    """Test that escalation requires exact persistence_count threshold"""
    camera_id = "cam_persistence"
    persistence_count = settings.persistence_count  # Default: 3
    high_threshold = settings.wildfire_high_threshold

    # Store (persistence_count - 1) detections
    for i in range(persistence_count - 1):
        detection = create_detection(confidence=high_threshold)
        level = await alert_manager.determine_alert_level([detection], camera_id=camera_id)
        assert level == "high", f"Detection {i+1} should be 'high', got '{level}'"
        await asyncio.sleep(0.1)

    # Next detection should trigger escalation
    level_final = await alert_manager.determine_alert_level([detection], camera_id=camera_id)
    assert level_final == "critical", \
        f"Detection {persistence_count} should escalate to 'critical', got '{level_final}'"

    print(f"✅ Escalation requires exactly {persistence_count} detections")


# ============================================================================
# Alert De-escalation Tests
# ============================================================================

@pytest.mark.asyncio
async def test_manual_deescalation_via_clear(clean_database):
    """Test manual de-escalation by clearing camera history"""
    camera_id = "cam_clear_test"

    # Build up to critical
    high_detection = create_detection(confidence=0.8)
    for i in range(3):
        await alert_manager.determine_alert_level([high_detection], camera_id=camera_id)
        await asyncio.sleep(0.1)

    # Verify critical state
    status = await alert_manager.get_camera_status(camera_id)
    assert status["current_alert_level"] == "critical"

    # Clear detections
    await alert_manager.clear_camera_detections(camera_id, hours=1)

    # Verify cleared
    records = await db_manager.get_detections_since(camera_id, minutes=60)
    assert len(records) == 0, f"Expected no records after clear, got {len(records)}"

    # New detection should start fresh at "high"
    level = await alert_manager.determine_alert_level([high_detection], camera_id=camera_id)
    assert level == "high", f"After clear, should be 'high', got '{level}'"

    print("✅ Manual de-escalation via clear_camera_detections works")


@pytest.mark.asyncio
async def test_temporal_window_deescalation(clean_database):
    """Test that old detections outside time window don't affect alert level"""
    camera_id = "cam_temporal_window"

    # This test simulates time passage by directly manipulating database timestamps
    # Store 3 high-confidence detections (would normally escalate to critical)
    for i in range(3):
        await db_manager.store_detection(
            camera_id=camera_id,
            confidence=0.8,
            detection_count=1
        )

    # Manually update timestamps to be outside escalation window
    # (This simulates detections from 4 hours ago)
    old_time = datetime.utcnow() - timedelta(hours=4)

    async with db_manager.get_connection() as conn:
        await conn.execute(
            "UPDATE camera_detections SET created_at = $1 WHERE camera_id = $2",
            old_time,
            camera_id
        )

    # New high-confidence detection should be "high" (not critical)
    # because old detections are outside 3-hour window
    detection = create_detection(confidence=0.8)
    level = await alert_manager.determine_alert_level([detection], camera_id=camera_id)

    assert level == "high", \
        f"Detection outside temporal window should be 'high', got '{level}'"

    print("✅ Temporal window correctly excludes old detections")


# ============================================================================
# Camera ID Temporal Tracking Tests
# ============================================================================

@pytest.mark.asyncio
async def test_camera_status_query(clean_database):
    """Test get_camera_status returns correct information"""
    camera_id = "cam_status_test"

    # Store varied detections
    confidences = [0.4, 0.5, 0.6, 0.8, 0.9]
    for conf in confidences:
        detection = create_detection(confidence=conf)
        await alert_manager.determine_alert_level([detection], camera_id=camera_id)
        await asyncio.sleep(0.1)

    # Query status
    status = await alert_manager.get_camera_status(camera_id)

    # Validate structure
    assert "camera_id" in status
    assert "current_alert_level" in status
    assert "recent_detections" in status
    assert "last_detection" in status
    assert "max_confidence" in status
    assert "confidence_breakdown" in status

    # Validate values
    assert status["camera_id"] == camera_id
    assert status["recent_detections"] == 5
    assert status["max_confidence"] == 0.9
    assert status["current_alert_level"] in ["none", "low", "high", "critical"]

    # Validate confidence breakdown
    breakdown = status["confidence_breakdown"]
    assert breakdown["high"] == 2  # 0.8, 0.9 are >= 0.7
    assert breakdown["medium"] == 3  # 0.4, 0.5, 0.6 are in [0.3, 0.7)
    assert breakdown["total"] == 5

    print(f"✅ Camera status query returns complete information")


@pytest.mark.asyncio
async def test_camera_first_detection(clean_database):
    """Test camera status on first detection"""
    camera_id = "cam_first_detection"

    # Store single detection
    detection = create_detection(confidence=0.5)
    level = await alert_manager.determine_alert_level([detection], camera_id=camera_id)

    status = await alert_manager.get_camera_status(camera_id)

    assert status["recent_detections"] == 1
    assert status["max_confidence"] == 0.5
    assert status["current_alert_level"] == "low"
    assert status["last_detection"] is not None

    print("✅ First detection correctly tracked")


@pytest.mark.asyncio
async def test_camera_no_history(clean_database):
    """Test camera status with no detection history"""
    camera_id = "cam_no_history"

    status = await alert_manager.get_camera_status(camera_id)

    assert status["current_alert_level"] == "none"
    assert status["recent_detections"] == 0
    assert status["last_detection"] is None
    assert status["max_confidence"] is None

    print("✅ Camera with no history returns 'none' state")


# ============================================================================
# Alert Lifecycle Tests
# ============================================================================

@pytest.mark.asyncio
async def test_complete_alert_lifecycle(clean_database):
    """Test complete alert lifecycle: detection -> escalation -> persistence -> de-escalation"""
    camera_id = "cam_lifecycle"

    # Phase 1: Initial detection (low)
    detection_medium = create_detection(confidence=0.5)
    level = await alert_manager.determine_alert_level([detection_medium], camera_id=camera_id)
    assert level == "low", f"Phase 1: Expected 'low', got '{level}'"
    await asyncio.sleep(0.1)

    # Phase 2: Persistence triggers escalation to high
    for i in range(2):
        level = await alert_manager.determine_alert_level([detection_medium], camera_id=camera_id)
        await asyncio.sleep(0.1)

    assert level == "high", f"Phase 2: Expected escalation to 'high', got '{level}'"

    # Phase 3: Upgrade to high-confidence detections
    detection_high = create_detection(confidence=0.8)
    level = await alert_manager.determine_alert_level([detection_high], camera_id=camera_id)
    assert level == "high", f"Phase 3: Expected 'high', got '{level}'"
    await asyncio.sleep(0.1)

    # Phase 4: High-confidence persistence triggers critical
    for i in range(2):
        level = await alert_manager.determine_alert_level([detection_high], camera_id=camera_id)
        await asyncio.sleep(0.1)

    assert level == "critical", f"Phase 4: Expected 'critical', got '{level}'"

    # Phase 5: Manual intervention (clear)
    await alert_manager.clear_camera_detections(camera_id)

    # Phase 6: Recovery - fresh start
    level = await alert_manager.determine_alert_level([detection_medium], camera_id=camera_id)
    assert level == "low", f"Phase 6: After clear, expected 'low', got '{level}'"

    print("✅ Complete alert lifecycle: low -> high -> critical -> clear -> low")


@pytest.mark.asyncio
async def test_intermittent_detection_pattern(clean_database):
    """Test alert behavior with intermittent detection pattern"""
    camera_id = "cam_intermittent"

    # Pattern: detect, pause, detect, pause, detect
    # Should still escalate if within time window
    detection = create_detection(confidence=0.8)

    levels = []
    for i in range(3):
        level = await alert_manager.determine_alert_level([detection], camera_id=camera_id)
        levels.append(level)
        await asyncio.sleep(0.2)  # Intermittent pattern

    # Should escalate to critical despite intermittent pattern
    assert levels[-1] == "critical", \
        f"Intermittent pattern should escalate, got {levels}"

    print(f"✅ Intermittent pattern escalates correctly: {' -> '.join(levels)}")


@pytest.mark.asyncio
async def test_confidence_threshold_boundaries(clean_database):
    """Test alert levels at exact threshold boundaries"""
    camera_id = "cam_boundaries"

    low_threshold = settings.wildfire_low_threshold  # 0.3
    high_threshold = settings.wildfire_high_threshold  # 0.7

    test_cases = [
        (low_threshold - 0.01, "none"),  # Just below low
        (low_threshold, "low"),           # Exactly at low
        (low_threshold + 0.01, "low"),    # Just above low
        (high_threshold - 0.01, "low"),   # Just below high
        (high_threshold, "high"),         # Exactly at high
        (high_threshold + 0.01, "high"),  # Just above high
    ]

    for confidence, expected_level in test_cases:
        # Use unique camera to avoid interference
        cam_id = f"{camera_id}_{confidence}"
        detection = create_detection(confidence=confidence)
        level = await alert_manager.determine_alert_level([detection], camera_id=cam_id)

        assert level == expected_level, \
            f"Confidence {confidence:.2f} should be '{expected_level}', got '{level}'"

    print("✅ Threshold boundaries correctly handled")


# ============================================================================
# System Statistics Tests
# ============================================================================

@pytest.mark.asyncio
async def test_system_statistics(clean_database):
    """Test system-wide statistics gathering"""
    # Create activity across multiple cameras
    cameras = ["cam1", "cam2", "cam3"]

    for cam_id in cameras:
        # Varied confidence levels
        confidences = [0.4, 0.6, 0.8]
        for conf in confidences:
            detection = create_detection(confidence=conf)
            await alert_manager.determine_alert_level([detection], camera_id=cam_id)
            await asyncio.sleep(0.05)

    # Get system stats
    stats = await alert_manager.get_system_stats()

    # Validate structure
    assert "confidence_distribution_24h" in stats
    assert "active_cameras" in stats
    assert "total_active_cameras" in stats
    assert "database_status" in stats

    # Validate counts
    assert stats["total_active_cameras"] == len(cameras)
    assert len(stats["active_cameras"]) == len(cameras)

    # Validate camera details
    for camera_info in stats["active_cameras"]:
        assert "camera_id" in camera_info
        assert "detection_count" in camera_info
        assert "max_confidence" in camera_info
        assert "avg_confidence" in camera_info
        assert "current_alert_level" in camera_info
        assert "last_detection" in camera_info

        assert camera_info["detection_count"] == 3
        assert camera_info["max_confidence"] == 0.8

    print(f"✅ System statistics: {stats['total_active_cameras']} active cameras")


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_concurrent_detections_same_camera(clean_database):
    """Test concurrent detections for same camera don't cause race conditions"""
    camera_id = "cam_concurrent"

    async def submit_detection(conf):
        detection = create_detection(confidence=conf)
        return await alert_manager.determine_alert_level([detection], camera_id=camera_id)

    # Submit 10 concurrent detections
    tasks = [submit_detection(0.5) for _ in range(10)]
    results = await asyncio.gather(*tasks)

    # All should complete without error
    assert len(results) == 10

    # Verify all stored in database
    records = await db_manager.get_detections_since(camera_id, minutes=10)
    assert len(records) == 10, f"Expected 10 records, got {len(records)}"

    print(f"✅ Concurrent operations handled correctly: {len(records)} records")


@pytest.mark.asyncio
async def test_database_unavailable_fallback(clean_database):
    """Test graceful fallback to basic mode when database unavailable"""
    # Close database connection
    await db_manager.close()

    # Reset initialization flag
    alert_manager.db_initialized = False

    # Should fall back to basic mode
    detection = create_detection(confidence=0.8)
    level = await alert_manager.determine_alert_level([detection], camera_id="cam_fallback")

    # Should still return correct level (basic mode)
    assert level == "high", f"Fallback mode should work, got '{level}'"

    # Reinitialize for other tests
    await db_manager.initialize()
    alert_manager.db_initialized = True

    print("✅ Graceful fallback to basic mode when database unavailable")


@pytest.mark.asyncio
async def test_multiple_detections_same_image(clean_database):
    """Test handling multiple smoke detections in single image"""
    camera_id = "cam_multi_detection"

    # Create multiple detections with different confidences
    detections = [
        create_detection(confidence=0.4, bbox=(100, 100, 200, 200)),
        create_detection(confidence=0.6, bbox=(300, 300, 400, 400)),
        create_detection(confidence=0.5, bbox=(500, 100, 600, 200)),
    ]

    # Should use max confidence (0.6)
    level = await alert_manager.determine_alert_level(detections, camera_id=camera_id)
    assert level == "low", f"Expected 'low' for max confidence 0.6, got '{level}'"

    # Verify stored with detection_count
    records = await db_manager.get_detections_since(camera_id, minutes=10)
    assert len(records) == 1
    assert records[0]["confidence"] == 0.6  # Max confidence

    print("✅ Multiple detections in single image handled correctly")


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests with pytest"""
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-s",  # Show print output
    ]

    return pytest.main(pytest_args)


async def run_tests_async():
    """Alternative async test runner (if pytest not available)"""
    print("=" * 80)
    print("SAI INFERENCE - ENHANCED ALERT MANAGEMENT TEST SUITE")
    print("=" * 80)

    # Initialize database once
    if not db_manager.pool:
        await db_manager.initialize()

    # Test groups
    test_groups = [
        ("Basic Alert Level Tests", [
            ("Basic mode (no camera_id)", test_basic_mode_no_camera_id),
            ("No detections", test_no_detections),
            ("Fire detections ignored", test_fire_detections_ignored),
        ]),
        ("Database Population Tests", [
            ("Database population", test_database_population),
            ("Temporal ordering", test_database_temporal_ordering),
            ("Multi-camera isolation", test_multiple_cameras_isolation),
        ]),
        ("Alert Escalation Tests", [
            ("High confidence escalation", test_high_confidence_escalation),
            ("Medium confidence escalation", test_medium_confidence_escalation),
            ("Persistence count", test_escalation_persistence_count),
        ]),
        ("Alert De-escalation Tests", [
            ("Manual de-escalation", test_manual_deescalation_via_clear),
            ("Temporal window", test_temporal_window_deescalation),
        ]),
        ("Camera Tracking Tests", [
            ("Camera status query", test_camera_status_query),
            ("First detection", test_camera_first_detection),
            ("No history", test_camera_no_history),
        ]),
        ("Alert Lifecycle Tests", [
            ("Complete lifecycle", test_complete_alert_lifecycle),
            ("Intermittent pattern", test_intermittent_detection_pattern),
            ("Threshold boundaries", test_confidence_threshold_boundaries),
        ]),
        ("System Statistics", [
            ("System stats", test_system_statistics),
        ]),
        ("Edge Cases", [
            ("Concurrent detections", test_concurrent_detections_same_camera),
            ("Database fallback", test_database_unavailable_fallback),
            ("Multiple detections", test_multiple_detections_same_image),
        ]),
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for group_name, tests in test_groups:
        print(f"\n{'=' * 80}")
        print(f"{group_name}")
        print("=" * 80)

        for test_name, test_func in tests:
            total_tests += 1
            print(f"\n{test_name}:")

            # Clean database before test
            try:
                async with db_manager.get_connection() as conn:
                    await conn.execute("DELETE FROM camera_detections")
            except Exception as e:
                print(f"⚠️  Failed to clean database: {e}")

            try:
                # Run test
                await test_func(None)
                passed_tests += 1
            except AssertionError as e:
                print(f"❌ FAILED: {e}")
                failed_tests.append((test_name, str(e)))
            except Exception as e:
                print(f"❌ ERROR: {e}")
                failed_tests.append((test_name, f"Error: {e}"))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")

    if failed_tests:
        print("\n❌ FAILED TESTS:")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {error}")
        return 1
    else:
        print("\n🎉 ALL TESTS PASSED!")
        return 0

    # Cleanup
    await db_manager.close()


if __name__ == "__main__":
    # Try pytest first, fall back to async runner
    try:
        import pytest
        exit(run_all_tests())
    except ImportError:
        print("pytest not available, using async test runner")
        exit(asyncio.run(run_tests_async()))
