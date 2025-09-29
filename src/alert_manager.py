"""
Enhanced Alert Manager for SAI Inference
Clean query-time alert calculation for wildfire smoke detection
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
        camera_id: Optional[str] = None
    ) -> str:
        """
        Determine alert level based on smoke detections

        Args:
            detections: List of detection objects
            camera_id: Optional camera identifier for enhanced tracking

        Returns:
            Alert level: "none", "low", "high", "critical"
        """
        # Filter smoke-only detections (wildfire focus)
        smoke_detections = [d for d in detections if d.class_name == "smoke"]

        if not smoke_detections:
            return "none"

        max_confidence = max(d.confidence for d in smoke_detections)

        # Route to appropriate mode
        if camera_id:
            return await self._enhanced_alert_logic(
                smoke_detections, max_confidence, camera_id
            )
        else:
            return self._basic_alert_logic(max_confidence)

    def _basic_alert_logic(self, max_confidence: float) -> str:
        """
        Basic confidence-only alert logic (no temporal tracking)
        Compatible with enhanced mode alert levels
        """
        if max_confidence >= settings.wildfire_high_threshold:
            return "high"
        elif max_confidence >= settings.wildfire_low_threshold:
            return "low"
        else:
            return "none"

    async def _enhanced_alert_logic(
        self,
        smoke_detections: List[Detection],
        max_confidence: float,
        camera_id: str
    ) -> str:
        """
        Enhanced alert logic with temporal tracking and query-time calculation

        Logic flow:
        1. Store raw detection fact
        2. Query recent detections to calculate alert level:
           - High confidence (≥0.7): If 3+ in 3h → "critical", else "high"
           - Medium confidence (0.3-0.7): If 3+ in 30m → "high", else "low"
           - Low confidence (<0.3): "none"
        """
        await self.ensure_db_initialized()

        if not self.db_initialized:
            logger.warning(f"Database unavailable, using basic mode for {camera_id}")
            return self._basic_alert_logic(max_confidence)

        try:
            # Return none immediately for low confidence
            if max_confidence < settings.wildfire_low_threshold:
                return "none"

            # Store raw detection fact
            await db_manager.store_detection(
                camera_id=camera_id,
                confidence=max_confidence,
                detection_count=len(smoke_detections),
                metadata={
                    "timestamp": datetime.utcnow().isoformat(),
                    "class_detections": [d.class_name for d in smoke_detections]
                }
            )

            # Calculate alert level based on current and recent detections
            return await self._calculate_alert_level(camera_id, max_confidence)

        except Exception as e:
            logger.error(f"Enhanced alert logic failed for {camera_id}: {e}")
            return self._basic_alert_logic(max_confidence)

    async def _calculate_alert_level(self, camera_id: str, current_confidence: float) -> str:
        """
        Calculate alert level based on current detection and recent history

        Uses query-time calculation for consistent state
        """
        try:
            # Base level from current detection
            if current_confidence < settings.wildfire_low_threshold:
                return "none"

            # Check for critical escalation (high confidence + persistence)
            if current_confidence >= settings.wildfire_high_threshold:
                # Count high-confidence detections in last 3 hours
                high_count = await db_manager.count_detections_by_confidence(
                    camera_id=camera_id,
                    min_confidence=settings.wildfire_high_threshold,
                    minutes=settings.escalation_hours * 60
                )

                if high_count >= settings.persistence_count:
                    logger.warning(
                        f"CRITICAL: Camera {camera_id} - {high_count} high detections "
                        f"in {settings.escalation_hours}h (confidence: {current_confidence:.3f})"
                    )
                    return "critical"
                else:
                    logger.info(
                        f"HIGH: Camera {camera_id} - high confidence detection "
                        f"({high_count}/{settings.persistence_count}, confidence: {current_confidence:.3f})"
                    )
                    return "high"

            # Check for high escalation (medium confidence + persistence)
            else:  # medium confidence (0.3 <= confidence < 0.7)
                medium_count = await db_manager.count_detections_by_confidence(
                    camera_id=camera_id,
                    min_confidence=settings.wildfire_low_threshold,
                    minutes=settings.escalation_minutes
                )

                if medium_count >= settings.persistence_count:
                    logger.info(
                        f"HIGH: Camera {camera_id} - {medium_count} medium detections "
                        f"in {settings.escalation_minutes}m (confidence: {current_confidence:.3f})"
                    )
                    return "high"
                else:
                    logger.debug(
                        f"LOW: Camera {camera_id} - medium confidence detection "
                        f"({medium_count}/{settings.persistence_count}, confidence: {current_confidence:.3f})"
                    )
                    return "low"

        except Exception as e:
            logger.error(f"Alert level calculation failed for {camera_id}: {e}")
            return self._basic_alert_logic(current_confidence)

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