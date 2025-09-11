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
    # SAINet2.1 Reference Parameters (from inf_yolo11m_SAINet2.1.py)
    confidence_threshold: float = Field(default=0.15, alias="SAI_CONFIDENCE")  # Reference: conf=0.15
    iou_threshold: float = Field(default=0.45, alias="SAI_IOU_THRESHOLD")
    
    # SAINet2.1 Optimized Resolution (864px - production optimized)
    input_size: Union[int, Tuple[int, int]] = Field(default=864, alias="SAI_INPUT_SIZE")  # Supports int or (height, width)
    max_detections: int = Field(default=100, alias="SAI_MAX_DETECTIONS")
    
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
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",  # Allow extra fields from environment variables
        protected_namespaces=('settings_',)  # Change protected namespace to avoid conflicts
    )
    
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