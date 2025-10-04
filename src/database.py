"""
Database module for SAI Inference Detection Tracking
PostgreSQL-based detection storage with query-time alert calculation
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base

from .config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy model for reference
Base = declarative_base()

class CameraDetection(Base):
    """Raw detection record for query-time alert calculation"""
    __tablename__ = 'camera_detections'

    id = Column(Integer, primary_key=True)
    camera_id = Column(String(100), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    detection_count = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    detection_metadata = Column(Text, nullable=True)


class DatabaseManager:
    """Async PostgreSQL database manager for detection records"""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.database_url = settings.database_url

    async def initialize(self):
        """Initialize database connection pool and create tables"""
        try:
            # Disable SSL for localhost connections to avoid permission issues
            # For production remote databases, enable SSL with proper certificates
            ssl_required = 'localhost' not in self.database_url and '127.0.0.1' not in self.database_url

            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=30,
                ssl=ssl_required,  # Disable SSL for localhost, enable for remote
                server_settings={
                    'application_name': 'sai-inference-detections',
                }
            )

            await self.create_tables()
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    async def create_tables(self):
        """Create enhanced camera detections table with comprehensive logging"""
        # Check if table already exists
        async with self.pool.acquire() as conn:
            table_exists = await conn.fetchval(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'camera_detections')"
            )

            if table_exists:
                logger.info("Enhanced schema already exists, skipping creation")
                return

            # Table doesn't exist, create it from migration file
            migration_file = Path(__file__).parent.parent / 'migrations' / '001_enhanced_schema.sql'

            if migration_file.exists():
                migration_sql = migration_file.read_text()
                # Execute migration (skip comments and empty lines)
                statements = [s.strip() for s in migration_sql.split(';') if s.strip() and not s.strip().startswith('--')]
                for statement in statements:
                    if statement:
                        try:
                            await conn.execute(statement)
                        except Exception as e:
                            # Ignore "already exists" errors
                            if "already exists" not in str(e):
                                raise
                logger.info("Enhanced schema created successfully")
            else:
                logger.error("Migration file not found and table doesn't exist!")
                raise FileNotFoundError(f"Migration file not found: {migration_file}")

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connections closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            raise RuntimeError("Database not initialized")

        async with self.pool.acquire() as conn:
            yield conn

    async def store_detection(
        self,
        camera_id: str,
        request_id: str,
        detection_count: int,
        smoke_count: int,
        fire_count: int,
        max_confidence: float,
        avg_confidence: Optional[float],
        base_alert_level: str,
        final_alert_level: str,
        escalation_reason: Optional[str],
        detections: List[Dict[str, Any]],
        processing_time_ms: Optional[float] = None,
        model_inference_time_ms: Optional[float] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        model_version: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        detection_classes: Optional[List[int]] = None,
        source: Optional[str] = None,
        n8n_workflow_id: Optional[str] = None,
        n8n_execution_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store comprehensive detection record with full inference details

        Args:
            camera_id: Camera identifier
            request_id: Unique request identifier
            detection_count: Total number of detections
            smoke_count: Number of smoke detections
            fire_count: Number of fire detections
            max_confidence: Maximum confidence score
            avg_confidence: Average confidence score
            base_alert_level: Initial alert level before escalation
            final_alert_level: Final alert level after temporal analysis
            escalation_reason: Why alert changed (NULL if no change)
            detections: List of detection objects with bbox and confidence
            processing_time_ms: Total processing time
            model_inference_time_ms: YOLO inference time only
            image_width: Image width in pixels
            image_height: Image height in pixels
            model_version: Model version identifier
            confidence_threshold: Confidence threshold used
            iou_threshold: IOU threshold used
            detection_classes: Classes filter used [0], [1], or [0,1]
            source: Request source (n8n, api-direct, mosaic, cli)
            n8n_workflow_id: n8n workflow identifier
            n8n_execution_id: n8n execution identifier
            metadata: Additional flexible metadata

        Returns:
            Database row ID
        """
        import json

        async with self.get_connection() as conn:
            detection_id = await conn.fetchval(
                """
                INSERT INTO camera_detections (
                    camera_id, request_id, detection_count, smoke_count, fire_count,
                    max_confidence, avg_confidence, base_alert_level, final_alert_level,
                    escalation_reason, detections, processing_time_ms, model_inference_time_ms,
                    image_width, image_height, model_version, confidence_threshold,
                    iou_threshold, detection_classes, source, n8n_workflow_id,
                    n8n_execution_id, metadata
                )
                VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                    $14, $15, $16, $17, $18, $19, $20, $21, $22, $23
                )
                RETURNING id
                """,
                camera_id, request_id, detection_count, smoke_count, fire_count,
                max_confidence, avg_confidence, base_alert_level, final_alert_level,
                escalation_reason,
                json.dumps(detections) if detections else None,
                processing_time_ms, model_inference_time_ms,
                image_width, image_height, model_version,
                confidence_threshold, iou_threshold,
                detection_classes,
                source, n8n_workflow_id, n8n_execution_id,
                json.dumps(metadata) if metadata else None
            )

        logger.debug(
            f"Stored detection {detection_id} for camera {camera_id} "
            f"(base: {base_alert_level}, final: {final_alert_level}, "
            f"detections: {detection_count}, confidence: {max_confidence:.3f})"
        )
        return detection_id

    async def get_detections_since(
        self,
        camera_id: str,
        minutes: int = 180  # 3 hours default
    ) -> List[Dict[str, Any]]:
        """Get recent detections for camera within time window"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        async with self.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT id, camera_id, max_confidence, detection_count, created_at, metadata
                FROM camera_detections
                WHERE camera_id = $1
                  AND created_at >= $2
                ORDER BY created_at DESC
                """,
                camera_id, cutoff_time
            )

        return [dict(row) for row in rows]

    async def count_detections_by_confidence(
        self,
        camera_id: str,
        min_confidence: float,
        minutes: int = 30
    ) -> int:
        """Count recent detections above confidence threshold"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        async with self.get_connection() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM camera_detections
                WHERE camera_id = $1
                  AND max_confidence >= $2
                  AND created_at >= $3
                """,
                camera_id, min_confidence, cutoff_time
            )

        return count or 0

    async def get_max_confidence_since(
        self,
        camera_id: str,
        minutes: int = 30
    ) -> Optional[float]:
        """Get maximum confidence detection in time window"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        async with self.get_connection() as conn:
            max_confidence_val = await conn.fetchval(
                """
                SELECT MAX(max_confidence)
                FROM camera_detections
                WHERE camera_id = $1
                  AND created_at >= $2
                """,
                camera_id, cutoff_time
            )

        return max_confidence_val

    async def cleanup_old_detections(self, days: int = 365):
        """Clean up old detection records"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        async with self.get_connection() as conn:
            result = await conn.execute(
                """
                DELETE FROM camera_detections
                WHERE created_at < $1
                """,
                cutoff_time
            )

        logger.info(f"Cleaned up old detections: {result}")

    async def get_camera_stats(
        self,
        camera_id: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get detection statistics for camera"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        async with self.get_connection() as conn:
            stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_detections,
                    AVG(max_confidence) as avg_confidence,
                    MAX(max_confidence) as max_confidence,
                    MAX(created_at) as last_detection
                FROM camera_detections
                WHERE camera_id = $1
                  AND created_at >= $2
                """,
                camera_id, cutoff_time
            )

        if not stats or stats['total_detections'] == 0:
            return {
                "camera_id": camera_id,
                "total_detections": 0,
                "avg_confidence": None,
                "max_confidence": None,
                "last_detection": None
            }

        return {
            "camera_id": camera_id,
            "total_detections": stats['total_detections'],
            "avg_confidence": float(stats['avg_confidence']) if stats['avg_confidence'] else None,
            "max_confidence": float(stats['max_confidence']) if stats['max_confidence'] else None,
            "last_detection": stats['last_detection'].isoformat() if stats['last_detection'] else None
        }

    async def get_all_cameras(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get list of all cameras with recent activity"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        async with self.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    camera_id,
                    MAX(created_at) as last_detection,
                    COUNT(*) as detection_count_24h,
                    (SELECT final_alert_level
                     FROM camera_detections cd2
                     WHERE cd2.camera_id = cd1.camera_id
                     ORDER BY created_at DESC
                     LIMIT 1) as last_alert_level
                FROM camera_detections cd1
                WHERE created_at >= $1
                GROUP BY camera_id
                ORDER BY last_detection DESC
                """,
                cutoff_time
            )

        return [dict(row) for row in rows]

    async def get_detections_with_metadata(
        self,
        camera_id: str,
        minutes: int = 180,
        min_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get recent detections with full metadata including alert state"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        async with self.get_connection() as conn:
            if min_confidence is not None:
                rows = await conn.fetch(
                    """
                    SELECT id, camera_id, max_confidence as confidence, detection_count, created_at,
                           base_alert_level, final_alert_level,
                           (base_alert_level != final_alert_level) as escalated,
                           escalation_reason
                    FROM camera_detections
                    WHERE camera_id = $1
                      AND created_at >= $2
                      AND max_confidence >= $3
                    ORDER BY created_at DESC
                    """,
                    camera_id, cutoff_time, min_confidence
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, camera_id, max_confidence as confidence, detection_count, created_at,
                           base_alert_level, final_alert_level,
                           (base_alert_level != final_alert_level) as escalated,
                           escalation_reason
                    FROM camera_detections
                    WHERE camera_id = $1
                      AND created_at >= $2
                    ORDER BY created_at DESC
                    """,
                    camera_id, cutoff_time
                )

        return [dict(row) for row in rows]

    async def get_escalation_events(
        self,
        camera_id: Optional[str] = None,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get escalation events (detections where base_alert_level != final_alert_level)"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        async with self.get_connection() as conn:
            if camera_id:
                rows = await conn.fetch(
                    """
                    SELECT id, camera_id, created_at, final_alert_level,
                           escalation_reason, max_confidence as confidence
                    FROM camera_detections
                    WHERE camera_id = $1
                      AND created_at >= $2
                      AND base_alert_level != final_alert_level
                    ORDER BY created_at DESC
                    """,
                    camera_id, cutoff_time
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, camera_id, created_at, final_alert_level,
                           escalation_reason, max_confidence as confidence
                    FROM camera_detections
                    WHERE created_at >= $1
                      AND base_alert_level != final_alert_level
                    ORDER BY created_at DESC
                    """,
                    cutoff_time
                )

        return [dict(row) for row in rows]

    async def get_recent_alerts(
        self,
        limit: int = 100,
        camera_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent alerts (all detections ordered by time)"""
        async with self.get_connection() as conn:
            if camera_id:
                rows = await conn.fetch(
                    """
                    SELECT id, camera_id, max_confidence as confidence, detection_count, created_at,
                           base_alert_level, final_alert_level,
                           (base_alert_level != final_alert_level) as escalated,
                           escalation_reason
                    FROM camera_detections
                    WHERE camera_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    camera_id, limit
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, camera_id, max_confidence as confidence, detection_count, created_at,
                           base_alert_level, final_alert_level,
                           (base_alert_level != final_alert_level) as escalated,
                           escalation_reason
                    FROM camera_detections
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    limit
                )

        return [dict(row) for row in rows]

    async def get_alert_summary(
        self,
        hours: int = 24,
        camera_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get aggregated alert statistics"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        async with self.get_connection() as conn:
            # Get alert level distribution
            if camera_id:
                level_stats = await conn.fetch(
                    """
                    SELECT final_alert_level, COUNT(*) as count
                    FROM camera_detections
                    WHERE camera_id = $1 AND created_at >= $2
                    GROUP BY final_alert_level
                    """,
                    camera_id, cutoff_time
                )

                total_alerts = await conn.fetchval(
                    "SELECT COUNT(*) FROM camera_detections WHERE camera_id = $1 AND created_at >= $2",
                    camera_id, cutoff_time
                )

                escalated_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM camera_detections WHERE camera_id = $1 AND created_at >= $2 AND base_alert_level != final_alert_level",
                    camera_id, cutoff_time
                )

                cameras_active = 1 if total_alerts > 0 else 0
            else:
                level_stats = await conn.fetch(
                    """
                    SELECT final_alert_level, COUNT(*) as count
                    FROM camera_detections
                    WHERE created_at >= $1
                    GROUP BY final_alert_level
                    """,
                    cutoff_time
                )

                total_alerts = await conn.fetchval(
                    "SELECT COUNT(*) FROM camera_detections WHERE created_at >= $1",
                    cutoff_time
                )

                escalated_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM camera_detections WHERE created_at >= $1 AND base_alert_level != final_alert_level",
                    cutoff_time
                )

                cameras_active = await conn.fetchval(
                    "SELECT COUNT(DISTINCT camera_id) FROM camera_detections WHERE created_at >= $1",
                    cutoff_time
                )

        # Build response
        by_level = {row['final_alert_level']: row['count'] for row in level_stats}
        escalation_rate = (escalated_count / total_alerts * 100) if total_alerts > 0 else 0.0

        return {
            "total_alerts": total_alerts or 0,
            "by_level": by_level,
            "escalation_rate": escalation_rate,
            "cameras_active": cameras_active or 0
        }

    async def get_escalation_stats(
        self,
        hours: int = 24,
        camera_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get escalation statistics"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        async with self.get_connection() as conn:
            if camera_id:
                reason_stats = await conn.fetch(
                    """
                    SELECT escalation_reason, COUNT(*) as count
                    FROM camera_detections
                    WHERE camera_id = $1
                      AND created_at >= $2
                      AND base_alert_level != final_alert_level
                    GROUP BY escalation_reason
                    """,
                    camera_id, cutoff_time
                )

                camera_stats = await conn.fetch(
                    """
                    SELECT camera_id, COUNT(*) as count
                    FROM camera_detections
                    WHERE camera_id = $1
                      AND created_at >= $2
                      AND base_alert_level != final_alert_level
                    GROUP BY camera_id
                    """,
                    camera_id, cutoff_time
                )

                total_escalations = await conn.fetchval(
                    "SELECT COUNT(*) FROM camera_detections WHERE camera_id = $1 AND created_at >= $2 AND base_alert_level != final_alert_level",
                    camera_id, cutoff_time
                )

                avg_confidence = await conn.fetchval(
                    "SELECT AVG(max_confidence) FROM camera_detections WHERE camera_id = $1 AND created_at >= $2 AND base_alert_level != final_alert_level",
                    camera_id, cutoff_time
                )
            else:
                reason_stats = await conn.fetch(
                    """
                    SELECT escalation_reason, COUNT(*) as count
                    FROM camera_detections
                    WHERE created_at >= $1 AND base_alert_level != final_alert_level
                    GROUP BY escalation_reason
                    """,
                    cutoff_time
                )

                camera_stats = await conn.fetch(
                    """
                    SELECT camera_id, COUNT(*) as count
                    FROM camera_detections
                    WHERE created_at >= $1 AND base_alert_level != final_alert_level
                    GROUP BY camera_id
                    """,
                    cutoff_time
                )

                total_escalations = await conn.fetchval(
                    "SELECT COUNT(*) FROM camera_detections WHERE created_at >= $1 AND base_alert_level != final_alert_level",
                    cutoff_time
                )

                avg_confidence = await conn.fetchval(
                    "SELECT AVG(max_confidence) FROM camera_detections WHERE created_at >= $1 AND base_alert_level != final_alert_level",
                    cutoff_time
                )

        by_reason = {row['escalation_reason']: row['count'] for row in reason_stats}
        by_camera = {row['camera_id']: row['count'] for row in camera_stats}

        return {
            "total_escalations": total_escalations or 0,
            "by_reason": by_reason,
            "by_camera": by_camera,
            "avg_confidence": float(avg_confidence) if avg_confidence else 0.0
        }


# Global database manager instance
db_manager = DatabaseManager()