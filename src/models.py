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
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    """Response model for inference endpoint"""
    request_id: str
    timestamp: datetime
    processing_time_ms: float
    image_size: Dict[str, int]  # {"width": int, "height": int}
    detections: List[Detection]
    detection_count: int
    has_fire: bool
    has_smoke: bool
    confidence_scores: Dict[str, float]  # Average confidence per class
    
    # Image storage (content-addressed)
    image_hash: Optional[str] = Field(None, description="SHA256 hash of raw input image")
    image_path: Optional[str] = Field(None, description="Storage path (filesystem or IPFS)")
    
    annotated_image: Optional[str] = Field(None, description="Base64 encoded annotated image")
    alert_level: Optional[str] = Field(None, description="Alert level: none/low/medium/high/critical")
    active_classes: List[str] = Field(default_factory=list, description="Detected class names")
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
    
    def determine_alert_level(self) -> str:
        """Determine alert level based on detections"""
        if not self.data.detections:
            return "none"
        
        fire_count = sum(1 for d in self.data.detections if d.class_name == "fire")
        smoke_count = sum(1 for d in self.data.detections if d.class_name == "smoke")
        max_confidence = max((d.confidence for d in self.data.detections), default=0)
        
        if fire_count > 2 or (fire_count > 0 and max_confidence > 0.8):
            return "critical"
        elif fire_count > 0:
            return "high"
        elif smoke_count > 2 or (smoke_count > 0 and max_confidence > 0.7):
            return "medium"
        elif smoke_count > 0:
            return "low"
        return "none"


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


# ============================================================================
# Camera Analytics & Alert History Models
# ============================================================================

class CameraListItem(BaseModel):
    """Camera with recent activity summary"""
    camera_id: str
    last_detection: Optional[datetime] = None
    detection_count_24h: int = 0
    last_alert_level: Optional[str] = None


class CameraStats(BaseModel):
    """Detection statistics for a specific camera"""
    camera_id: str
    total_detections: int = 0
    avg_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    last_detection: Optional[str] = None


class DetectionRecord(BaseModel):
    """Single detection record with alert metadata"""
    id: int
    camera_id: str
    confidence: float
    detection_count: int
    captured_at: Optional[datetime] = None
    base_alert_level: str
    final_alert_level: str
    escalated: bool = False
    escalation_reason: Optional[str] = None


class EscalationEvent(BaseModel):
    """Escalation event record"""
    id: int
    camera_id: str
    captured_at: Optional[datetime] = None
    final_alert_level: str
    escalation_reason: Optional[str] = None
    confidence: float


class AlertSummary(BaseModel):
    """Aggregated alert statistics"""
    total_alerts: int = 0
    by_level: Dict[str, int] = Field(default_factory=dict)
    escalation_rate: float = 0.0
    cameras_active: int = 0


class EscalationStats(BaseModel):
    """Escalation statistics"""
    total_escalations: int = 0
    by_reason: Dict[str, int] = Field(default_factory=dict)
    by_camera: Dict[str, int] = Field(default_factory=dict)
    avg_confidence: float = 0.0