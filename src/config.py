"""
SAI Inference Service Configuration
"""
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
import os

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
    # SAINet2.1 Reference Parameters (from inf_yolo11m_SAINet2.1.py)
    confidence_threshold: float = Field(default=0.15, alias="SAI_CONFIDENCE")  # Reference: conf=0.15
    iou_threshold: float = Field(default=0.45, alias="SAI_IOU_THRESHOLD")
    
    # SAINet2.1 Optimized Resolution (1920px - from reference)
    input_size: int = Field(default=1920, alias="SAI_INPUT_SIZE")  # Reference: imgsz=1920
    max_detections: int = Field(default=100, alias="SAI_MAX_DETECTIONS")
    
    # Performance
    batch_size: int = Field(default=1, env="SAI_BATCH_SIZE")
    max_queue_size: int = Field(default=100, env="SAI_MAX_QUEUE")
    
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
    
    
    # File Upload
    max_upload_size: int = Field(default=50 * 1024 * 1024, env="SAI_MAX_UPLOAD")  # 50MB
    allowed_extensions: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
        env="SAI_ALLOWED_EXTENSIONS"
    )
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",  # Allow extra fields from environment variables
        protected_namespaces=('settings_',)  # Change protected namespace to avoid conflicts
    )


# Create settings instance - will load from .env and environment
settings = Settings()