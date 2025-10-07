"""
SAI Inference Service Configuration
"""
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict, field_validator
from pathlib import Path
from typing import Optional, List, Union, Tuple
from dotenv import load_dotenv
import os
import json

# Force load .env file before settings initialization
load_dotenv(override=True)


class Settings(BaseSettings):
    # API Configuration
    app_name: str = "SAI Inference Service"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    host: str = Field(default="0.0.0.0", alias="SAI_HOST")
    port: int = Field(default=8888, alias="SAI_PORT")
    workers: int = Field(default=1, alias="SAI_WORKERS")

    # Model Configuration
    models_dir: Path = Field(default=Path("models"), alias="SAI_MODEL_DIR")
    default_model: str = Field(default="sai_v2.1.pt", alias="SAI_DEFAULT_MODEL")
    device: str = Field(default="cpu", alias="SAI_DEVICE")  # cpu, cuda, cuda:0
    # Production optimized parameters (matches deployment/production.env)
    confidence_threshold: float = Field(default=0.39, alias="SAI_CONFIDENCE_THRESHOLD")  # Production optimized
    iou_threshold: float = Field(default=0.1, alias="SAI_IOU_THRESHOLD")  # Production optimized (lower = more overlapping boxes allowed)

    # SAINet2.1 Optimized Resolution (864px - production optimized)
    input_size: Union[int, Tuple[int, int]] = Field(default=864, alias="SAI_INPUT_SIZE")  # Supports int or (height, width)
    max_detections: int = Field(default=100, alias="SAI_MAX_DETECTIONS")

    # Detection Classes Filter - Default to smoke-only for wildfire early warning
    default_detection_classes: Optional[List[int]] = Field(
        default=[0],
        alias="SAI_DETECTION_CLASSES",
        description="Filter to specific class IDs (0=smoke, 1=fire). Default=[0] for smoke-only wildfire detection"
    )

    # Performance
    batch_size: int = Field(default=1, env="SAI_BATCH_SIZE")
    max_queue_size: int = Field(default=100, env="SAI_MAX_QUEUE")

    # n8n Integration
    n8n_api_key: Optional[str] = Field(default=None, env="SAI_API_KEY")
    allowed_origins: List[str] = Field(
        default=["*"],
        env="SAI_ALLOWED_ORIGINS"
    )

    # Monitoring
    enable_metrics: bool = Field(default=True, env="SAI_ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="SAI_METRICS_PORT")
    log_level: str = Field(default="INFO", env="SAI_LOG_LEVEL")


    # File Upload
    max_upload_size: int = Field(default=50 * 1024 * 1024, env="SAI_MAX_UPLOAD")  # 50MB
    allowed_extensions: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
        env="SAI_ALLOWED_EXTENSIONS"
    )

    # Enhanced Alert System (Wildfire Detection)
    database_url: str = Field(
        default="postgresql://sai_user:password@localhost/sai_inference",
        env="SAI_DATABASE_URL"
    )
    wildfire_high_threshold: float = Field(
        default=0.7,
        env="SAI_WILDFIRE_HIGH_THRESHOLD",
        description="Confidence threshold for immediate high alert"
    )
    wildfire_low_threshold: float = Field(
        default=0.3,
        env="SAI_WILDFIRE_LOW_THRESHOLD",
        description="Confidence threshold for temporal tracking"
    )
    escalation_hours: int = Field(
        default=3,
        env="SAI_ESCALATION_HOURS",
        description="Hours to track for critical escalation"
    )
    escalation_minutes: int = Field(
        default=30,
        env="SAI_ESCALATION_MINUTES",
        description="Minutes to track for high escalation"
    )
    persistence_count: int = Field(
        default=3,
        env="SAI_PERSISTENCE_COUNT",
        description="Number of detections needed for escalation"
    )

    model_config = ConfigDict(
        # Note: .env file is loaded if present (for development)
        # Production should use environment variables via systemd (no .env file)
        env_file=".env",
        env_ignore_empty=True,  # Ignore if .env doesn't exist
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",  # Allow extra fields from environment variables
        protected_namespaces=('settings_',)  # Change protected namespace to avoid conflicts
    )

    @field_validator('default_detection_classes', mode='before')
    @classmethod
    def parse_detection_classes(cls, v):
        """Parse detection_classes from environment variable string to list of integers.

        Supports:
        - None or empty: None (detect both classes)
        - Single class: "0" → [0] (smoke-only)
        - Multiple classes: "0,1" → [0, 1] (both)
        - JSON format: "[0]" → [0]
        """
        if v is None or v == "" or v == "null":
            return None

        if isinstance(v, str):
            v = v.strip()

            # Try JSON format first: "[0]" or "[0,1]"
            if v.startswith('[') and v.endswith(']'):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [int(x) for x in parsed]
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

            # Try comma-separated format: "0" or "0,1"
            try:
                return [int(x.strip()) for x in v.split(',') if x.strip()]
            except (ValueError, AttributeError):
                pass

        return v  # Return as-is if already a list or other type

    @field_validator('input_size', mode='before')
    @classmethod
    def parse_input_size(cls, v):
        """Parse input_size from various string formats in environment variables.

        Supports:
        - Integer: 864 → 864
        - Tuple formats: (480, 640), [480, 640], 480,640 → (480, 640)
        """
        if isinstance(v, str):
            v = v.strip()

            # Try to parse tuple format: (480, 640)
            if v.startswith('(') and v.endswith(')'):
                try:
                    inner = v[1:-1].strip()
                    parts = [int(x.strip()) for x in inner.split(',')]
                    if len(parts) == 2:
                        return tuple(parts)
                except (ValueError, AttributeError):
                    pass

            # Try to parse list format: [480, 640]
            elif v.startswith('[') and v.endswith(']'):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list) and len(parsed) == 2:
                        return tuple(int(x) for x in parsed)
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

            # Try comma-separated format: 480,640
            elif ',' in v:
                try:
                    parts = [int(x.strip()) for x in v.split(',')]
                    if len(parts) == 2:
                        return tuple(parts)
                except (ValueError, AttributeError):
                    pass

        # Return as-is for normal int parsing or other types
        return v


# Create settings instance - will load from .env and environment
settings = Settings()