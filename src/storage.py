"""
SAI Image Storage Module

Content-addressed storage for raw inference images.
Designed for seamless IPFS migration in Phase 2.

Storage layout:
  {base_path}/{hash[0:2]}/{hash[2:4]}/{hash}.jpg

Usage:
  from .storage import image_storage
  result = await image_storage.store(image_bytes)
  image_bytes = await image_storage.fetch(result.hash)
"""

import hashlib
import aiofiles
import aiofiles.os
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import io
import os

logger = logging.getLogger(__name__)


@dataclass
class StorageResult:
    """Result of storing an image"""
    hash: str           # SHA256 hex digest (64 chars)
    path: str           # Filesystem path
    size: int           # Size in bytes
    is_duplicate: bool  # True if already existed (dedup)
    stored_at: datetime


class ImageStorage:
    """
    Content-addressed image storage.

    Interface designed for easy IPFS migration:
    - store(bytes) -> StorageResult with hash
    - fetch(hash) -> bytes
    - exists(hash) -> bool
    - get_url(hash) -> URL for HTTP access
    """

    def __init__(
        self,
        base_path: Optional[str] = None,
        shard_depth: int = 2,
        shard_width: int = 2,
    ):
        # Default to env var or ./data/images (development-friendly default)
        self.base_path = Path(
            base_path or
            os.environ.get('SAI_IMAGE_STORAGE_PATH', './data/images')
        )
        self.shard_depth = shard_depth
        self.shard_width = shard_width

        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Image storage initialized: {self.base_path}")

    def _compute_hash(self, data: bytes) -> str:
        """Compute SHA256 hash of image data"""
        return hashlib.sha256(data).hexdigest()

    def _get_shard_path(self, image_hash: str) -> Path:
        """Get sharded directory path for a hash"""
        parts = []
        for i in range(self.shard_depth):
            start = i * self.shard_width
            end = start + self.shard_width
            parts.append(image_hash[start:end])
        return self.base_path.joinpath(*parts)

    def _get_file_path(self, image_hash: str, extension: str = ".jpg") -> Path:
        """Get full file path for a hash"""
        shard_dir = self._get_shard_path(image_hash)
        return shard_dir / f"{image_hash}{extension}"

    async def store(
        self,
        image_data: Union[bytes, io.BytesIO],
        extension: str = ".jpg",
    ) -> StorageResult:
        """
        Store image with content-based addressing.

        Args:
            image_data: Raw image bytes or BytesIO
            extension: File extension (default .jpg)

        Returns:
            StorageResult with hash, path, size, dedup status
        """
        if isinstance(image_data, io.BytesIO):
            image_data = image_data.getvalue()

        image_hash = self._compute_hash(image_data)
        file_path = self._get_file_path(image_hash, extension)

        is_duplicate = file_path.exists()

        if not is_duplicate:
            shard_dir = self._get_shard_path(image_hash)
            await aiofiles.os.makedirs(shard_dir, exist_ok=True)

            # Atomic write: temp file then rename
            temp_path = file_path.with_suffix('.tmp')
            try:
                async with aiofiles.open(temp_path, 'wb') as f:
                    await f.write(image_data)
                await aiofiles.os.rename(temp_path, file_path)
                logger.info(f"Stored: {image_hash[:16]}... ({len(image_data):,} bytes)")
            except Exception as e:
                if temp_path.exists():
                    await aiofiles.os.remove(temp_path)
                logger.error(f"Storage failed: {e}")
                raise
        else:
            logger.debug(f"Deduplicated: {image_hash[:16]}...")

        return StorageResult(
            hash=image_hash,
            path=str(file_path),
            size=len(image_data),
            is_duplicate=is_duplicate,
            stored_at=datetime.now(timezone.utc)
        )

    async def fetch(self, image_hash: str, extension: str = ".jpg") -> Optional[bytes]:
        """Fetch image bytes by hash"""
        file_path = self._get_file_path(image_hash, extension)

        if not file_path.exists():
            logger.warning(f"Not found: {image_hash[:16]}...")
            return None

        async with aiofiles.open(file_path, 'rb') as f:
            return await f.read()

    def exists(self, image_hash: str, extension: str = ".jpg") -> bool:
        """Check if image exists by hash"""
        return self._get_file_path(image_hash, extension).exists()

    def get_path(self, image_hash: str, extension: str = ".jpg") -> str:
        """Get filesystem path for hash (doesn't check existence)"""
        return str(self._get_file_path(image_hash, extension))

    def get_relative_path(self, image_hash: str, extension: str = ".jpg") -> str:
        """Get path relative to base_path"""
        file_path = self._get_file_path(image_hash, extension)
        return str(file_path.relative_to(self.base_path))

    async def delete(self, image_hash: str, extension: str = ".jpg") -> bool:
        """Delete image by hash. Returns True if deleted."""
        file_path = self._get_file_path(image_hash, extension)
        if not file_path.exists():
            return False
        await aiofiles.os.remove(file_path)
        logger.info(f"Deleted: {image_hash[:16]}...")
        return True


# Global singleton
image_storage = ImageStorage()
