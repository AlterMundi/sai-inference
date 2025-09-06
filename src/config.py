"""
SAI Inference Service Configuration
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Optional, List
import os


class Settings(BaseSettings):
    # API Configuration
    app_name: str = "SAI Inference Service"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    host: str = Field(default="0.0.0.0", env="SAI_HOST")
    port: int = Field(default=8888, env="SAI_PORT")
    workers: int = Field(default=1, env="SAI_WORKERS")
    
    # Model Configuration
    model_dir: Path = Field(default=Path("models"), env="SAI_MODEL_DIR")
    default_model: str = Field(default="sai_v2.1.pt", env="SAI_DEFAULT_MODEL")
    model_device: str = Field(default="cpu", env="SAI_DEVICE")  # cpu, cuda, cuda:0
    model_confidence: float = Field(default=0.45, env="SAI_CONFIDENCE")
    model_iou_threshold: float = Field(default=0.45, env="SAI_IOU_THRESHOLD")
    
    # SACRED Resolution (896x896)
    input_size: int = Field(default=896, env="SAI_INPUT_SIZE")
    max_detections: int = Field(default=100, env="SAI_MAX_DETECTIONS")
    
    # Performance
    batch_size: int = Field(default=1, env="SAI_BATCH_SIZE")
    max_queue_size: int = Field(default=100, env="SAI_MAX_QUEUE")
    cache_enabled: bool = Field(default=True, env="SAI_CACHE_ENABLED")
    cache_ttl: int = Field(default=300, env="SAI_CACHE_TTL")  # seconds
    
    # n8n Integration
    n8n_webhook_path: str = Field(default="/webhook/sai", env="SAI_WEBHOOK_PATH")
    n8n_api_key: Optional[str] = Field(default=None, env="SAI_API_KEY")
    allowed_origins: List[str] = Field(
        default=["*"],
        env="SAI_ALLOWED_ORIGINS"
    )
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="SAI_ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="SAI_METRICS_PORT")
    log_level: str = Field(default="INFO", env="SAI_LOG_LEVEL")
    
    # Redis Cache (optional)
    redis_url: Optional[str] = Field(default=None, env="SAI_REDIS_URL")
    
    # File Upload
    max_upload_size: int = Field(default=50 * 1024 * 1024, env="SAI_MAX_UPLOAD")  # 50MB
    allowed_extensions: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
        env="SAI_ALLOWED_EXTENSIONS"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()