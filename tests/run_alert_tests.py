#!/usr/bin/env python3
"""
Simple test runner for alert management tests
Avoids pytest-asyncio event loop issues
"""
import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alert_manager import alert_manager
from src.database import db_manager
from src.models import Detection, BoundingBox, DetectionClass
from src.config import settings


def create_detection(class_name="smoke", confidence=0.5, bbox=(100, 100, 200, 200)):
    """Create a test detection"""
    return Detection(
        class_name=DetectionClass.SMOKE if class_name == "smoke" else DetectionClass.FIRE,
        class_id=0 if class_name == "smoke" else 1,
        confidence=confidence,
        bbox=BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
    )


async def clean_db():
    """Clean database"""
    async with db_manager.get_connection() as conn:
        await conn.execute("DELETE FROM camera_detections")


# ==================================================================
# Tests
# ==================================================================

async def test_basic_mode():
    """Test basic alert mode without camera_id"""
    await clean_db()

    # Low confidence -> "none"
    level = await alert_manager.determine_alert_level([create_detection(confidence=0.2)], camera_id=None)
    assert level == "none", f"Expected 'none', got '{level}'"

    # Medium -> "low"
    level = await alert_manager.determine_alert_level([create_detection(confidence=0.4)], camera_id=None)
    assert level == "low", f"Expected 'low', got '{level}'"

    # High -> "high"
    level = await alert_manager.determine_alert_level([create_detection(confidence=0.8)], camera_id=None)
    assert level == "high", f"Expected 'high', got '{level}'"

    print("✅ Basic mode works correctly")


async def test_database_population():
    """Test database population"""
    await clean_db()

    camera_id = "test_cam_001"
    detection = create_detection(confidence=0.6)

    for i in range(5):
        await alert_manager.determine_alert_level([detection], camera_id=camera_id)
        await asyncio.sleep(0.05)

    records = await db_manager.get_detections_since(camera_id, minutes=10)
    assert len(records) == 5, f"Expected 5 records, got {len(records)}"

    for record in records:
        assert record["camera_id"] == camera_id
        assert 0 <= record["confidence"] <= 1

    print(f"✅ Database correctly stores {len(records)} records")


async def test_high_confidence_escalation():
    """Test high confidence escalation to critical"""
    await clean_db()

    camera_id = "cam_escalation"
    detection = create_detection(confidence=0.8)

    level1 = await alert_manager.determine_alert_level([detection], camera_id=camera_id)
    assert level1 == "high", f"First detection should be 'high', got '{level1}'"
    await asyncio.sleep(0.1)

    level2 = await alert_manager.determine_alert_level([detection], camera_id=camera_id)
    assert level2 == "high", f"Second detection should be 'high', got '{level2}'"
    await asyncio.sleep(0.1)

    level3 = await alert_manager.determine_alert_level([detection], camera_id=camera_id)
    assert level3 == "critical", f"Third detection should be 'critical', got '{level3}'"

    print("✅ High confidence escalation: high → high → critical")


async def test_medium_confidence_escalation():
    """Test medium confidence escalation"""
    await clean_db()

    camera_id = "cam_medium"
    detection = create_detection(confidence=0.5)

    level1 = await alert_manager.determine_alert_level([detection], camera_id=camera_id)
    assert level1 == "low", f"First detection should be 'low', got '{level1}'"
    await asyncio.sleep(0.1)

    level2 = await alert_manager.determine_alert_level([detection], camera_id=camera_id)
    assert level2 == "low", f"Second detection should be 'low', got '{level2}'"
    await asyncio.sleep(0.1)

    level3 = await alert_manager.determine_alert_level([detection], camera_id=camera_id)
    assert level3 == "high", f"Third detection should be 'high', got '{level3}'"

    print("✅ Medium confidence escalation: low → low → high")


async def test_camera_isolation():
    """Test camera isolation"""
    await clean_db()

    cameras = ["cam_A", "cam_B", "cam_C"]

    for idx, cam_id in enumerate(cameras):
        num_detections = (idx + 1) * 2  # 2, 4, 6
        for i in range(num_detections):
            detection = create_detection(confidence=0.5)
            await alert_manager.determine_alert_level([detection], camera_id=cam_id)
            await asyncio.sleep(0.05)

    for idx, cam_id in enumerate(cameras):
        expected = (idx + 1) * 2
        records = await db_manager.get_detections_since(cam_id, minutes=10)
        assert len(records) == expected, f"Camera {cam_id}: expected {expected}, got {len(records)}"
        for rec in records:
            assert rec["camera_id"] == cam_id

    print(f"✅ {len(cameras)} cameras correctly isolated")


async def test_camera_status():
    """Test camera status query"""
    await clean_db()

    camera_id = "cam_status"
    confidences = [0.4, 0.5, 0.6, 0.8, 0.9]

    for conf in confidences:
        detection = create_detection(confidence=conf)
        await alert_manager.determine_alert_level([detection], camera_id=camera_id)
        await asyncio.sleep(0.1)

    status = await alert_manager.get_camera_status(camera_id)

    assert status["camera_id"] == camera_id
    assert status["recent_detections"] == 5
    assert status["max_confidence"] == 0.9
    assert status["current_alert_level"] in ["none", "low", "high", "critical"]

    breakdown = status["confidence_breakdown"]
    assert breakdown["high"] == 2  # 0.8, 0.9
    assert breakdown["medium"] == 3  # 0.4, 0.5, 0.6
    assert breakdown["total"] == 5

    print("✅ Camera status query returns complete information")


async def test_manual_clear():
    """Test manual de-escalation"""
    await clean_db()

    camera_id = "cam_clear"
    detection = create_detection(confidence=0.8)

    # Build to critical
    for i in range(3):
        await alert_manager.determine_alert_level([detection], camera_id=camera_id)
        await asyncio.sleep(0.1)

    status = await alert_manager.get_camera_status(camera_id)
    assert status["current_alert_level"] == "critical"

    # Clear
    await alert_manager.clear_camera_detections(camera_id, hours=1)

    records = await db_manager.get_detections_since(camera_id, minutes=60)
    assert len(records) == 0

    # New detection starts fresh
    level = await alert_manager.determine_alert_level([detection], camera_id=camera_id)
    assert level == "high"

    print("✅ Manual de-escalation via clear_camera_detections works")


async def test_concurrent_detections():
    """Test concurrent detections"""
    await clean_db()

    camera_id = "cam_concurrent"

    async def submit_detection(conf):
        detection = create_detection(confidence=conf)
        return await alert_manager.determine_alert_level([detection], camera_id=camera_id)

    tasks = [submit_detection(0.5) for _ in range(10)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 10

    records = await db_manager.get_detections_since(camera_id, minutes=10)
    assert len(records) == 10

    print(f"✅ Concurrent operations: {len(records)} records stored correctly")


async def test_system_stats():
    """Test system statistics"""
    await clean_db()

    cameras = ["cam1", "cam2", "cam3"]

    for cam_id in cameras:
        confidences = [0.4, 0.6, 0.8]
        for conf in confidences:
            detection = create_detection(confidence=conf)
            await alert_manager.determine_alert_level([detection], camera_id=cam_id)
            await asyncio.sleep(0.05)

    stats = await alert_manager.get_system_stats()

    assert "confidence_distribution_24h" in stats
    assert "active_cameras" in stats
    assert "total_active_cameras" in stats

    assert stats["total_active_cameras"] == len(cameras)
    assert len(stats["active_cameras"]) == len(cameras)

    for camera_info in stats["active_cameras"]:
        assert camera_info["detection_count"] == 3
        assert camera_info["max_confidence"] == 0.8

    print(f"✅ System statistics: {stats['total_active_cameras']} active cameras")


# ==================================================================
# Test Runner
# ==================================================================

async def main():
    """Run all tests"""
    print("=" * 80)
    print("SAI INFERENCE - ENHANCED ALERT MANAGEMENT TEST SUITE")
    print("=" * 80)

    # Initialize database
    if not db_manager.pool:
        await db_manager.initialize()
        print(f"✓ Database initialized: {settings.database_url}")

    tests = [
        ("Basic Alert Mode", test_basic_mode),
        ("Database Population", test_database_population),
        ("High Confidence Escalation", test_high_confidence_escalation),
        ("Medium Confidence Escalation", test_medium_confidence_escalation),
        ("Camera Isolation", test_camera_isolation),
        ("Camera Status Query", test_camera_status),
        ("Manual De-escalation", test_manual_clear),
        ("Concurrent Detections", test_concurrent_detections),
        ("System Statistics", test_system_stats),
    ]

    passed = 0
    failed = []

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))

        try:
            await test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed.append((test_name, str(e)))
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed.append((test_name, f"Error: {e}"))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(failed)}")
    print(f"Success rate: {passed/len(tests)*100:.1f}%")

    if failed:
        print("\n❌ FAILED TESTS:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        return 1
    else:
        print("\n🎉 ALL TESTS PASSED!")
        return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
