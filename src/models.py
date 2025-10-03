"""
Data models for SAI Inference Service
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class DetectionClass(str, Enum):
    """SAI Detection Classes"""
    SMOKE = "smoke"
    FIRE = "fire"


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_xywh(self) -> Dict[str, float]:
        """Convert to x,y,width,height format"""
        return {
            "x": self.x1,
            "y": self.y1,
            "width": self.width,
            "height": self.height
        }
    
    def to_normalized(self, img_width: int, img_height: int) -> "BoundingBox":
        """Normalize coordinates to 0-1 range"""
        return BoundingBox(
            x1=self.x1 / img_width,
            y1=self.y1 / img_height,
            x2=self.x2 / img_width,
            y2=self.y2 / img_height
        )


class Detection(BaseModel):
    """Single detection result"""
    class_name: DetectionClass
    class_id: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    model_config = ConfigDict(use_enum_values=True, protected_namespaces=())


class InferenceRequest(BaseModel):
    """Request model for inference endpoint"""
    # Image Input
    image: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL to download image from")
    
    # Core Detection Parameters
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence for detections")
    iou_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="IOU threshold for NMS")
    max_detections: Optional[int] = Field(None, ge=1, le=1000, description="Maximum detections per image")
    
    # High-Value YOLO Parameters (Official Ultralytics)
    detection_classes: Optional[List[int]] = Field(
        None, 
        description="Filter to specific class IDs (0=smoke, 1=fire, None=both)",
        examples=[[0], [1], [0,1], None]
    )
    half_precision: bool = Field(
        False, 
        description="Enable FP16 inference for 2x speed boost (requires compatible GPU)"
    )
    test_time_augmentation: bool = Field(
        False, 
        description="Enable TTA for improved accuracy (2-3x slower inference)"
    )
    class_agnostic_nms: bool = Field(
        False, 
        description="Suppress overlapping detections across fire/smoke classes"
    )
    
    # Annotation Control
    return_image: bool = Field(False, description="Return annotated image")
    show_labels: bool = Field(True, description="Include class labels in annotations")
    show_confidence: bool = Field(True, description="Display confidence scores in annotations")
    line_width: Optional[int] = Field(None, ge=1, le=10, description="Bounding box line thickness")
    
    # Processing Options
    webhook_url: Optional[str] = Field(None, description="Webhook URL for async processing")
    camera_id: Optional[str] = Field(None, description="Camera identifier for enhanced temporal alert tracking")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    """Response model for inference endpoint"""
    request_id: str
    timestamp: datetime
    processing_time_ms: float
    image_size: Dict[str, int]  # {"width": int, "height": int}
    detections: List[Detection]
    detection_count: int

    # Wildfire Detection Results
    has_fire: bool
    has_smoke: bool
    confidence_scores: Dict[str, float]  # Average confidence per class

    # Enhanced Wildfire Alert System
    alert_level: Optional[str] = Field(None, description="Wildfire alert level: none, low, high, critical")
    detection_mode: Optional[str] = Field(None, description="Detection mode: smoke-only, fire-only, both")
    active_classes: Optional[List[str]] = Field(None, description="Currently active detection classes")
    camera_id: Optional[str] = Field(None, description="Camera identifier (echo back for n8n verification)")

    annotated_image: Optional[str] = Field(None, description="Base64 encoded annotated image")
    version: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(use_enum_values=True, protected_namespaces=())


class BatchInferenceRequest(BaseModel):
    """Request for batch inference"""
    images: List[Union[str, Dict[str, Any]]]  # Base64 or URLs
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    iou_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_detections_per_image: Optional[int] = Field(None, ge=1, le=1000)
    return_images: bool = Field(False)
    parallel_processing: bool = Field(True)
    camera_ids: Optional[List[Optional[str]]] = Field(None, description="Camera IDs for each image (same order as images)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BatchInferenceResponse(BaseModel):
    """Response for batch inference"""
    request_id: str
    timestamp: datetime
    total_processing_time_ms: float
    results: List[InferenceResponse]
    total_detections: int
    summary: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    version: str
    path: str
    size_mb: float
    classes: List[str]
    input_size: int
    confidence_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    device: str
    loaded: bool
    performance_metrics: Optional[Dict[str, float]] = None


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    is_model_loaded: bool
    loaded_model_info: Optional[ModelInfo] = None
    system_metrics: Dict[str, Any]
    runtime_parameters: Optional[Dict[str, Any]] = None


class WebhookPayload(BaseModel):
    """Webhook payload for n8n"""
    event_type: str = "detection"
    timestamp: datetime
    source: str = "sai-inference"
    data: InferenceResponse
    alert_level: Optional[str] = None  # "low", "medium", "high", "critical"
    
    async def determine_alert_level(self, camera_id: Optional[str] = None) -> str:
        """
        Determine alert level using enhanced alert manager

        Args:
            camera_id: Optional camera identifier for temporal tracking

        Returns:
            Alert level: "none", "low", "high", "critical"
        """
        from .alert_manager import alert_manager

        return await alert_manager.determine_alert_level(
            detections=self.data.detections,
            camera_id=camera_id
        )


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None