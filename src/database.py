"""
Database module for SAI Inference Detection Tracking
PostgreSQL-based detection storage with query-time alert calculation
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

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
    metadata = Column(Text, nullable=True)


class DatabaseManager:
    """Async PostgreSQL database manager for detection records"""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.database_url = settings.database_url

    async def initialize(self):
        """Initialize database connection pool and create tables"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=30,
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
        """Create camera detections table if it doesn't exist"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS camera_detections (
            id SERIAL PRIMARY KEY,
            camera_id VARCHAR(100) NOT NULL,
            confidence FLOAT NOT NULL,
            detection_count INTEGER DEFAULT 1 CHECK (detection_count > 0),
            created_at TIMESTAMP DEFAULT NOW(),
            metadata TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_camera_detections_camera_id ON camera_detections(camera_id);
        CREATE INDEX IF NOT EXISTS idx_camera_detections_created_at ON camera_detections(created_at);
        CREATE INDEX IF NOT EXISTS idx_camera_detections_camera_time ON camera_detections(camera_id, created_at);
        """

        async with self.pool.acquire() as conn:
            await conn.execute(create_table_sql)

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
        confidence: float,
        detection_count: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Store new detection record"""
        metadata_json = str(metadata) if metadata else None

        async with self.get_connection() as conn:
            detection_id = await conn.fetchval(
                """
                INSERT INTO camera_detections
                (camera_id, confidence, detection_count, metadata)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                camera_id, confidence, detection_count, metadata_json
            )

        logger.debug(f"Stored detection {detection_id} for camera {camera_id} (confidence: {confidence})")
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
                SELECT id, camera_id, confidence, detection_count, created_at, metadata
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
                  AND confidence >= $2
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
            max_confidence = await conn.fetchval(
                """
                SELECT MAX(confidence)
                FROM camera_detections
                WHERE camera_id = $1
                  AND created_at >= $2
                """,
                camera_id, cutoff_time
            )

        return max_confidence

    async def cleanup_old_detections(self, days: int = 7):
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
                    AVG(confidence) as avg_confidence,
                    MAX(confidence) as max_confidence,
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
            "avg_confidence": float(stats['avg_confidence']),
            "max_confidence": float(stats['max_confidence']),
            "last_detection": stats['last_detection'].isoformat() if stats['last_detection'] else None
        }


# Global database manager instance
db_manager = DatabaseManager()