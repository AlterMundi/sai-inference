"""
Inference Context - Data structure for comprehensive detection logging
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from .models import Detection


@dataclass
class InferenceContext:
    """
    Comprehensive context for inference execution logging
    Passed to alert_manager for database storage
    """
    # Required fields
    request_id: str
    camera_id: str
    detections: List[Detection]
    captured_at: datetime = None  # Actual capture time from camera node (UTC)

    # Detection summary
    detection_count: int = 0
    smoke_count: int = 0
    fire_count: int = 0
    max_confidence: float = 0.0
    avg_confidence: Optional[float] = None

    # Performance metrics
    processing_time_ms: Optional[float] = None
    model_inference_time_ms: Optional[float] = None

    # Image metadata
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    # Model configuration
    model_version: Optional[str] = None
    confidence_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    detection_classes: Optional[List[int]] = None

    # Request context
    source: str = "api-direct"  # 'n8n', 'api-direct', 'mosaic', 'cli'
    n8n_workflow_id: Optional[str] = None
    n8n_execution_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_detection_dict_list(self) -> List[Dict[str, Any]]:
        """Convert Detection objects to dict list for JSONB storage"""
        return [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox": {
                    "x1": d.bbox.x1,
                    "y1": d.bbox.y1,
                    "x2": d.bbox.x2,
                    "y2": d.bbox.y2
                }
            }
            for d in self.detections
        ]
