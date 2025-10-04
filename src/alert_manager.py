"""
Enhanced Alert Manager for SAI Inference
Clean query-time alert calculation for wildfire smoke detection

ALERT SYSTEM ARCHITECTURE:
==========================

Two-Mode Operation:
-------------------
1. Basic Mode (no camera_id):
   - Single detection analysis
   - No database logging
   - Returns: "none", "low", "high"

2. Enhanced Mode (with camera_id):
   - Temporal pattern analysis
   - Database logging of ALL executions
   - Returns: "none", "low", "high", "critical"

Single-Source Base Alert Level:
--------------------------------
Both modes use the same base judgment logic:
- No detections → "none"
- Detection confidence >= 0.7 → "high"
- Detection confidence < 0.7 → "low"

Enhanced Temporal Escalation:
------------------------------
Uses dual-window strategy for optimal wildfire detection:

1. Low → High Escalation (30-minute window):
   - Detects short-term smoke persistence
   - 3+ detections in 30 minutes → escalate to "high"
   - Quick de-escalation when pattern stops (within 30m)

2. High → Critical Escalation (3-hour window):
   - Detects long-term high-confidence patterns
   - 3+ high-confidence (>=0.7) in 3 hours → escalate to "critical"
   - Slower de-escalation for sustained threats

Window Strategy Rationale:
---------------------------
- Short window (30m) for low confidence: Catches immediate threats while allowing quick recovery from false positives
- Long window (3h) for high confidence: Tracks serious sustained threats, appropriate for wildfire progression timescales
- Separate windows prevent premature de-escalation of serious threats while avoiding alert fatigue from transient smoke

Edge Cases:
-----------
- Zero detections: Always "none", temporal history ignored
- Database failure: Falls back to base level (no escalation)
- Time window boundaries: Detections outside window are not counted (natural de-escalation)
- Multiple detections in single request: All count toward persistence threshold
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from .config import settings
from .database import db_manager
from .models import Detection

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages alert level determination with dual-mode operation:
    - Basic Mode: Simple confidence-based alerts (no camera_id)
    - Enhanced Mode: Temporal tracking with query-time calculation (with camera_id)
    """

    def __init__(self):
        self.db_initialized = False

    async def ensure_db_initialized(self):
        """Ensure database is initialized for enhanced mode"""
        if not self.db_initialized:
            try:
                await db_manager.initialize()
                self.db_initialized = True
                logger.info("Alert manager database initialized")
            except Exception as e:
                logger.error(f"Failed to initialize alert database: {e}")

    async def determine_alert_level(
        self,
        detections: List[Detection],
        camera_id: Optional[str] = None,
        context: Optional['InferenceContext'] = None
    ) -> str:
        """
        Determine alert level based on smoke detections

        Args:
            detections: List of detection objects
            camera_id: Optional camera identifier for enhanced tracking
            context: Optional full inference context for comprehensive logging

        Returns:
            Alert level: "none", "low", "high", "critical"
        """
        # Get base alert level (single source of truth)
        base_level, max_confidence, smoke_detections = self._get_base_alert_level(detections)

        # Route to enhanced mode if camera_id provided (logs ALL executions + temporal analysis)
        if camera_id:
            return await self._enhanced_alert_logic(
                base_level, max_confidence, smoke_detections, camera_id, context
            )

        # Basic mode: return base level directly (no logging, no temporal analysis)
        return base_level

    def _get_base_alert_level(self, detections: List[Detection]) -> tuple[str, float, List[Detection]]:
        """
        Single source of truth for base alert level determination
        Used by BOTH basic and enhanced modes

        Business Logic:
        - No detections → "none"
        - Detection confidence >= 0.7 → "high"
        - Detection confidence < 0.7 → "low"

        Returns:
            (alert_level, max_confidence, smoke_detections)
        """
        # Filter smoke-only detections (wildfire focus)
        smoke_detections = [d for d in detections if d.class_name == "smoke"]

        if not smoke_detections:
            return ("none", 0.0, smoke_detections)

        max_confidence = max(d.confidence for d in smoke_detections)

        # Simple two-tier system: high (>=0.7) or low (<0.7)
        if max_confidence >= settings.wildfire_high_threshold:
            alert_level = "high"
        else:
            alert_level = "low"

        return (alert_level, max_confidence, smoke_detections)

    async def _enhanced_alert_logic(
        self,
        base_level: str,
        max_confidence: float,
        smoke_detections: List[Detection],
        camera_id: str,
        context: Optional['InferenceContext'] = None
    ) -> str:
        """
        Enhanced alert logic with temporal tracking and escalation/de-escalation

        IMPORTANT: Logs ALL executions when camera_id is provided (even zero detections)

        Logic flow:
        1. Store raw detection fact (uses base_level from single source)
        2. Apply temporal escalation rules:
           - "none" → stays "none" (no escalation from zero)
           - "low" + persistence (3+ in 30m) → escalate to "high"
           - "high" + persistence (3+ in 3h) → escalate to "critical"
        """
        await self.ensure_db_initialized()

        if not self.db_initialized:
            logger.warning(f"Database unavailable, returning base level for {camera_id}")
            return base_level

        try:
            # No escalation for "none" base level
            if base_level == "none":
                # Log zero-detection execution with full context
                await self._store_detection_with_context(
                    camera_id=camera_id,
                    base_level="none",
                    final_level="none",
                    escalation_reason=None,
                    max_confidence=max_confidence,
                    smoke_detections=smoke_detections,
                    context=context
                )
                return "none"

            # Apply temporal escalation based on base level
            final_level, escalation_reason = await self._apply_temporal_escalation(
                base_level, camera_id, max_confidence
            )

            # Store detection with full context
            await self._store_detection_with_context(
                camera_id=camera_id,
                base_level=base_level,
                final_level=final_level,
                escalation_reason=escalation_reason,
                max_confidence=max_confidence,
                smoke_detections=smoke_detections,
                context=context
            )

            return final_level

        except Exception as e:
            logger.error(f"Enhanced alert logic failed for {camera_id}: {e}")
            return base_level  # Fallback to base level on error

    async def _store_detection_with_context(
        self,
        camera_id: str,
        base_level: str,
        final_level: str,
        escalation_reason: Optional[str],
        max_confidence: float,
        smoke_detections: List[Detection],
        context: Optional['InferenceContext'] = None
    ):
        """Store detection with full inference context if available"""
        # Count smoke vs fire detections from context if available
        all_detections = context.detections if context else smoke_detections
        smoke_count = sum(1 for d in all_detections if d.class_name == "smoke")
        fire_count = sum(1 for d in all_detections if d.class_name == "fire")

        # Calculate average confidence
        avg_conf = None
        if all_detections:
            avg_conf = sum(d.confidence for d in all_detections) / len(all_detections)

        # Use context data if available, otherwise use minimal data
        await db_manager.store_detection(
            camera_id=camera_id,
            request_id=context.request_id if context else "unknown",
            detection_count=len(all_detections),
            smoke_count=smoke_count,
            fire_count=fire_count,
            max_confidence=max_confidence,
            avg_confidence=avg_conf,
            base_alert_level=base_level,
            final_alert_level=final_level,
            escalation_reason=escalation_reason,
            detections=context.to_detection_dict_list() if context else [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox": {
                        "x1": d.bbox.x1,
                        "y1": d.bbox.y1,
                        "x2": d.bbox.x2,
                        "y2": d.bbox.y2
                    }
                }
                for d in smoke_detections
            ],
            processing_time_ms=context.processing_time_ms if context else None,
            model_inference_time_ms=context.model_inference_time_ms if context else None,
            image_width=context.image_width if context else None,
            image_height=context.image_height if context else None,
            model_version=context.model_version if context else None,
            confidence_threshold=context.confidence_threshold if context else None,
            iou_threshold=context.iou_threshold if context else None,
            detection_classes=context.detection_classes if context else None,
            source=context.source if context else "api-direct",
            n8n_workflow_id=context.n8n_workflow_id if context else None,
            n8n_execution_id=context.n8n_execution_id if context else None,
            metadata=context.metadata if context else None
        )

    async def _apply_temporal_escalation(
        self,
        base_level: str,
        camera_id: str,
        current_confidence: float
    ) -> tuple[str, Optional[str]]:
        """
        Apply temporal escalation/de-escalation based on detection patterns

        Escalation Rules:
        - "low" base + 3+ detections in 30m → escalate to "high"
        - "high" base + 3+ detections in 3h → escalate to "critical"

        Natural De-escalation:
        When detections stop or become sporadic, alerts automatically de-escalate
        as old detections age out of their respective time windows:
        - "high" (from low escalation) → "low" when count drops below 3 in 30m window
        - "critical" → "high" when high-confidence count drops below 3 in 3h window

        Example Timeline (30m window for low confidence):
        T-35m: detection → counted at T-35m, not counted at T-0m (outside window)
        T-10m: detection → counted at T-10m and T-0m
        T-0m:  detection → if total in 30m window >= 3, escalates to "high"
        T+35m: detection → T-35m aged out, may de-escalate if count < 3

        Args:
            base_level: Base alert level from single source ("low" or "high")
            camera_id: Camera identifier for history lookup
            current_confidence: Current detection confidence

        Returns:
            (final_alert_level, escalation_reason)
            escalation_reason: "persistence_high", "persistence_low", or None
        """
        try:
            if base_level == "high":
                # Check for critical escalation (high confidence + persistence)
                high_count = await db_manager.count_detections_by_confidence(
                    camera_id=camera_id,
                    min_confidence=settings.wildfire_high_threshold,
                    minutes=settings.escalation_hours * 60
                )

                if high_count >= settings.persistence_count:
                    logger.warning(
                        f"CRITICAL ESCALATION: Camera {camera_id} - {high_count} high detections "
                        f"in {settings.escalation_hours}h (confidence: {current_confidence:.3f})"
                    )
                    return ("critical", "persistence_high")
                else:
                    logger.info(
                        f"HIGH (base): Camera {camera_id} - high confidence detection "
                        f"({high_count}/{settings.persistence_count}, confidence: {current_confidence:.3f})"
                    )
                    return ("high", None)

            elif base_level == "low":
                # Check for high escalation (low confidence + persistence)
                # Count ANY detections (no minimum confidence) in recent window
                low_count = await db_manager.count_detections_by_confidence(
                    camera_id=camera_id,
                    min_confidence=0.0,  # Count all detections
                    minutes=settings.escalation_minutes
                )

                if low_count >= settings.persistence_count:
                    logger.info(
                        f"HIGH ESCALATION: Camera {camera_id} - {low_count} persistent detections "
                        f"in {settings.escalation_minutes}m (confidence: {current_confidence:.3f})"
                    )
                    return ("high", "persistence_low")
                else:
                    logger.debug(
                        f"LOW (base): Camera {camera_id} - low confidence detection "
                        f"({low_count}/{settings.persistence_count}, confidence: {current_confidence:.3f})"
                    )
                    return ("low", None)

            # Should never reach here, but return base level as fallback
            return (base_level, None)

        except Exception as e:
            logger.error(f"Temporal escalation failed for {camera_id}: {e}")
            return (base_level, None)  # Fallback to base level on error

    async def get_camera_status(self, camera_id: str) -> Dict[str, Any]:
        """
        Get current alert status for a camera using query-time calculation
        """
        await self.ensure_db_initialized()

        if not self.db_initialized:
            return {"error": "Database unavailable"}

        try:
            # Get recent detections and stats
            recent_detections = await db_manager.get_detections_since(
                camera_id=camera_id,
                minutes=settings.escalation_hours * 60
            )

            if not recent_detections:
                return {
                    "camera_id": camera_id,
                    "current_alert_level": "none",
                    "recent_detections": 0,
                    "last_detection": None,
                    "max_confidence": None
                }

            # Calculate current alert level from most recent detection
            latest_detection = recent_detections[0]
            max_confidence = latest_detection["confidence"]

            # Use same logic as determine_alert_level but without storing new detection
            current_level = await self._calculate_alert_level(camera_id, max_confidence)

            # Get confidence statistics
            high_confidence_count = sum(
                1 for d in recent_detections
                if d["confidence"] >= settings.wildfire_high_threshold
            )
            medium_confidence_count = sum(
                1 for d in recent_detections
                if settings.wildfire_low_threshold <= d["confidence"] < settings.wildfire_high_threshold
            )

            return {
                "camera_id": camera_id,
                "current_alert_level": current_level,
                "recent_detections": len(recent_detections),
                "last_detection": latest_detection["created_at"].isoformat(),
                "max_confidence": max_confidence,
                "confidence_breakdown": {
                    "high": high_confidence_count,
                    "medium": medium_confidence_count,
                    "total": len(recent_detections)
                }
            }

        except Exception as e:
            logger.error(f"Failed to get camera status for {camera_id}: {e}")
            return {"error": str(e)}

    async def clear_camera_detections(self, camera_id: str, hours: int = 24):
        """
        Clear recent detection history for a camera
        (e.g., false alarm confirmed, maintenance reset)
        """
        await self.ensure_db_initialized()

        if self.db_initialized:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)

                async with db_manager.get_connection() as conn:
                    result = await conn.execute(
                        """
                        DELETE FROM camera_detections
                        WHERE camera_id = $1 AND created_at >= $2
                        """,
                        camera_id, cutoff_time
                    )

                logger.info(f"Cleared {hours}h detection history for camera {camera_id}: {result}")
            except Exception as e:
                logger.error(f"Failed to clear detections for {camera_id}: {e}")

    async def cleanup_old_data(self):
        """
        Periodic cleanup of old detection data
        """
        await self.ensure_db_initialized()

        if self.db_initialized:
            try:
                # Clean up old detection records (7 days default)
                await db_manager.cleanup_old_detections(days=7)
                logger.info("Completed detection data cleanup")
            except Exception as e:
                logger.error(f"Detection data cleanup failed: {e}")

    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system-wide detection statistics
        """
        await self.ensure_db_initialized()

        if not self.db_initialized:
            return {"error": "Database unavailable"}

        try:
            # Get recent activity across all cameras
            cutoff_time = datetime.utcnow() - timedelta(hours=24)

            async with db_manager.get_connection() as conn:
                # Detection confidence distribution in last 24h
                confidence_stats = await conn.fetch(
                    """
                    SELECT
                        CASE
                            WHEN confidence >= $2 THEN 'high'
                            WHEN confidence >= $3 THEN 'medium'
                            ELSE 'low'
                        END as confidence_level,
                        COUNT(*) as count,
                        AVG(confidence) as avg_confidence
                    FROM camera_detections
                    WHERE created_at >= $1
                    GROUP BY confidence_level
                    ORDER BY avg_confidence DESC
                    """,
                    cutoff_time, settings.wildfire_high_threshold, settings.wildfire_low_threshold
                )

                # Active cameras with their current calculated alert levels
                active_cameras = await conn.fetch(
                    """
                    SELECT camera_id,
                           COUNT(*) as detection_count,
                           MAX(confidence) as max_confidence,
                           AVG(confidence) as avg_confidence,
                           MAX(created_at) as last_detection
                    FROM camera_detections
                    WHERE created_at >= $1
                    GROUP BY camera_id
                    ORDER BY last_detection DESC
                    """,
                    cutoff_time
                )

            # Calculate current alert levels for each camera
            cameras_with_alerts = []
            for camera in active_cameras:
                # Calculate alert level for this camera
                current_level = await self._calculate_alert_level(
                    camera["camera_id"],
                    camera["max_confidence"]
                )

                cameras_with_alerts.append({
                    "camera_id": camera["camera_id"],
                    "detection_count": camera["detection_count"],
                    "max_confidence": float(camera["max_confidence"]),
                    "avg_confidence": float(camera["avg_confidence"]),
                    "current_alert_level": current_level,
                    "last_detection": camera["last_detection"].isoformat()
                })

            return {
                "confidence_distribution_24h": {
                    row["confidence_level"]: {
                        "count": row["count"],
                        "avg_confidence": float(row["avg_confidence"])
                    } for row in confidence_stats
                },
                "active_cameras": cameras_with_alerts,
                "total_active_cameras": len(cameras_with_alerts),
                "database_status": "connected"
            }

        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {"error": str(e)}


# Global alert manager instance
alert_manager = AlertManager()