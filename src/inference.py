"""
SAI Inference Engine with YOLO model management
"""
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import cv2
import base64
import io
from PIL import Image
import time
import logging
from datetime import datetime

from .config import settings
from .models import (
    Detection, BoundingBox, DetectionClass,
    InferenceResponse, ModelInfo
)

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages YOLO model loading and switching"""
    
    def __init__(self):
        self.models: Dict[str, YOLO] = {}
        self.current_model: Optional[YOLO] = None
        self.current_model_name: Optional[str] = None
        self.model_info: Dict[str, ModelInfo] = {}
        self.device = self._setup_device()
        
    def _setup_device(self) -> str:
        """Setup compute device"""
        device = settings.device.lower()
        
        if device.startswith("cuda"):
            if torch.cuda.is_available():
                if ":" in device:
                    return device
                return "cuda:0"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
        return "cpu"
    
    def discover_models(self) -> List[str]:
        """Discover available model files in models directory"""
        model_extensions = ['.pt', '.pth', '.onnx', '.engine']
        models = []
        
        models_dir = Path(settings.models_dir)
        if not models_dir.exists():
            logger.warning(f"Models directory does not exist: {models_dir}")
            return []
            
        for model_file in models_dir.iterdir():
            if model_file.is_file() and model_file.suffix.lower() in model_extensions:
                models.append(model_file.name)
        
        logger.info(f"Discovered {len(models)} model(s): {models}")
        return sorted(models)
    
    def get_best_available_model(self) -> Optional[str]:
        """Get the best available model to load"""
        available_models = self.discover_models()
        if not available_models:
            return None
            
        # Priority order: configured default > SAI models > any .pt file
        default_model = settings.default_model
        if default_model in available_models:
            return default_model
            
        # Look for SAI/sai models
        sai_models = [m for m in available_models if 'sai' in m.lower()]
        if sai_models:
            return sai_models[0]
            
        # Look for YOLOv8/11 models
        yolo_models = [m for m in available_models if any(x in m.lower() for x in ['yolo', 'yv8', 'yv11'])]
        if yolo_models:
            return yolo_models[0]
            
        # Return first .pt file
        pt_models = [m for m in available_models if m.endswith('.pt')]
        if pt_models:
            return pt_models[0]
            
        return available_models[0] if available_models else None
    
    def load_model(self, model_name: str, model_path: Optional[Path] = None) -> bool:
        """Load a YOLO model"""
        try:
            if model_path is None:
                model_path = settings.models_dir / model_name
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            logger.info(f"Loading model: {model_name} from {model_path}")
            model = YOLO(str(model_path))
            
            # Move to device
            if self.device != "cpu":
                model.to(self.device)
            
            self.models[model_name] = model
            
            # Store model info
            self.model_info[model_name] = ModelInfo(
                name=model_name,
                version="SAI-v2.1",
                path=str(model_path),
                size_mb=model_path.stat().st_size / (1024 * 1024),
                classes=["smoke", "fire"],
                input_size=settings.input_size,  # Dynamic from settings
                confidence_threshold=settings.confidence_threshold,  # Dynamic from settings
                iou_threshold=settings.iou_threshold,  # Dynamic from settings
                device=self.device,
                loaded=True
            )
            
            # Set as current if no model is active
            if self.current_model is None:
                self.set_current_model(model_name)
            
            logger.info(f"Model {model_name} loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def set_current_model(self, model_name: str) -> bool:
        """Set the current active model"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not loaded")
            return False
        
        self.current_model = self.models[model_name]
        self.current_model_name = model_name
        logger.info(f"Active model set to: {model_name}")
        return True
    
    def get_model_info(self, model_name: Optional[str] = None) -> Optional[ModelInfo]:
        """Get information about a model"""
        if model_name is None:
            model_name = self.current_model_name
        
        if model_name and model_name in self.model_info:
            return self.model_info[model_name]
        return None
    
    def list_models(self) -> List[str]:
        """List all loaded models"""
        return list(self.models.keys())


class InferenceEngine:
    """Main inference engine for SAI detection"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        # CACHE COMPLETELY REMOVED - was causing identical outputs
        # self.cache: Dict[str, InferenceResponse] = {}
        # self.cache_timestamps: Dict[str, float] = {}
        
        # Load default model
        self._load_default_model()
    
    def _load_default_model(self):
        """Load the best available model on startup"""
        # Discover available models
        available_models = self.model_manager.discover_models()
        if not available_models:
            logger.warning("No models found in models directory")
            return
            
        # Try to load the best available model
        best_model = self.model_manager.get_best_available_model()
        if best_model:
            logger.info(f"Loading best available model: {best_model}")
            success = self.model_manager.load_model(best_model)
            if success:
                return
                
        # Fallback: try each available model
        logger.warning(f"Default model {settings.default_model} failed, trying alternatives")
        for model_name in available_models:
            logger.info(f"Attempting to load: {model_name}")
            if self.model_manager.load_model(model_name):
                logger.info(f"Successfully loaded fallback model: {model_name}")
                return
                
        logger.error("Failed to load any available models")
    
    def _copy_development_model(self):
        """Copy model from development directory"""
        try:
            import shutil
            settings.models_dir.mkdir(parents=True, exist_ok=True)
            
            # Look for model in local models directory
            local_model = Path("models/last.pt")
            if local_model.exists():
                dest = settings.models_dir / "sai_v2.1.pt"
                shutil.copy2(local_model, dest)
                logger.info(f"Copied local model to {dest}")
                return
            
            # No model found
            logger.warning("No model found in models/last.pt - please add a model file")
                
        except Exception as e:
            logger.error(f"Failed to copy development model: {e}")
    
    
    def _decode_image_direct(self, image_data: str):
        """Direct base64 to PIL with health checks"""
        try:
            # Remove data URL prefix if present
            if "," in image_data:
                image_data = image_data.split(",")[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            return self._validate_and_prepare_image(image, "base64_string")
        except Exception as e:
            logger.error(f"Base64 image decoding failed: {e}")
            raise
    
    def _validate_and_prepare_image(self, image: Image.Image, source: str = "unknown") -> Image.Image:
        """Health check and prepare image for YOLO processing with proper logging"""
        
        # Log image properties
        logger.info(f"Image health check - Source: {source}, Size: {image.size}, Mode: {image.mode}, Format: {getattr(image, 'format', 'Unknown')}")
        
        # Basic validation without verify() which consumes the image
        try:
            # Check if image has valid size (this validates it's properly loaded)
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions: {width}x{height}")
        except Exception as e:
            logger.error(f"Image validation failed - Source: {source}, Error: {e}")
            raise ValueError(f"Corrupted or invalid image from {source}: {e}")
        
        # Check image dimensions
        width, height = image.size
        if width < 32 or height < 32:
            logger.error(f"Image too small - Source: {source}, Size: {width}x{height}, Minimum: 32x32")
            raise ValueError(f"Image too small: {width}x{height}. Minimum size is 32x32 pixels")
        
        if width > 8192 or height > 8192:
            logger.warning(f"Very large image - Source: {source}, Size: {width}x{height}, May cause memory issues")
        
        # RGB conversion with logging
        if image.mode != "RGB":
            logger.info(f"Converting image mode - Source: {source}, From: {image.mode}, To: RGB")
            try:
                image = image.convert("RGB")
                logger.info(f"Image mode conversion successful - Source: {source}")
            except Exception as e:
                logger.error(f"Image mode conversion failed - Source: {source}, Error: {e}")
                raise ValueError(f"Failed to convert image to RGB from {source}: {e}")
        else:
            logger.debug(f"Image already in RGB mode - Source: {source}")
        
        # Final validation
        if image.mode != "RGB":
            logger.error(f"Image mode validation failed - Source: {source}, Final mode: {image.mode}")
            raise ValueError(f"Image processing failed: expected RGB, got {image.mode}")
        
        logger.info(f"Image health check passed - Source: {source}, Final size: {image.size}")
        return image
    
    def _decode_image_binary(self, image_bytes: bytes):
        """Direct binary bytes to PIL - optimal path with health checks"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return self._validate_and_prepare_image(image, "binary_bytes")
        except Exception as e:
            logger.error(f"Binary image decoding failed: {e}")
            raise
    
    # CACHE METHODS COMPLETELY REMOVED TO FIX IDENTICAL OUTPUT BUG
    # def _generate_cache_key(self, image_data: str, params: Dict[str, Any]) -> str:
    #     """Generate cache key for inference request"""
    #     key_data = f"{image_data[:100]}{json.dumps(params, sort_keys=True)}"
    #     return hashlib.md5(key_data.encode()).hexdigest()
    # 
    # def _check_cache(self, cache_key: str) -> Optional[InferenceResponse]:
    #     """Check if result is in cache"""
    #     if not settings.cache_enabled:
    #         return None
    #     
    #     if cache_key in self.cache:
    #         timestamp = self.cache_timestamps.get(cache_key, 0)
    #         if time.time() - timestamp < settings.cache_ttl:
    #             logger.debug(f"Cache hit for key: {cache_key}")
    #             return self.cache[cache_key]
    #         else:
    #             # Expired
    #             del self.cache[cache_key]
    #             del self.cache_timestamps[cache_key]
    #     
    #     return None
    # 
    # def _update_cache(self, cache_key: str, response: InferenceResponse):
    #     """Update cache with new result"""
    #     if settings.cache_enabled:
    #         self.cache[cache_key] = response
    #         self.cache_timestamps[cache_key] = time.time()
    #         
    #         # Cleanup old entries if cache is too large
    #         if len(self.cache) > 1000:
    #             # Remove oldest 100 entries
    #             sorted_keys = sorted(
    #                 self.cache_timestamps.keys(),
    #                 key=lambda k: self.cache_timestamps[k]
    #             )
    #             for key in sorted_keys[:100]:
    #                 del self.cache[key]
    #                 del self.cache_timestamps[key]
    
    async def infer(
        self,
        image_data: Union[str, bytes, np.ndarray],
        request_id: str,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        return_annotated: bool = False,
        # High-Value YOLO Parameters
        detection_classes: Optional[List[int]] = None,
        half_precision: bool = False,
        test_time_augmentation: bool = False,
        class_agnostic_nms: bool = False,
        # Annotation Control
        show_labels: bool = True,
        show_confidence: bool = True,
        line_width: Optional[int] = None,
        # Enhanced Alert System
        camera_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> InferenceResponse:
        """Run inference on an image"""
        
        start_time = time.time()
        
        # Use defaults if not provided
        confidence = confidence_threshold or settings.confidence_threshold
        iou = iou_threshold or settings.iou_threshold
        max_det = max_detections or settings.max_detections
        detection_classes = detection_classes or settings.default_detection_classes
        
        # Validate detection_classes parameter
        if detection_classes is not None:
            if not all(0 <= cls <= 1 for cls in detection_classes):
                raise ValueError("detection_classes must contain only valid class IDs: 0 (smoke) or 1 (fire)")
            if len(detection_classes) == 0:
                raise ValueError("detection_classes cannot be empty")
        
        # Check GPU availability for half_precision
        if half_precision and not torch.cuda.is_available():
            logger.warning("Half precision requested but CUDA not available, falling back to FP32")
            half_precision = False
        
        try:
            # Handle different input types optimally
            if isinstance(image_data, str):
                # Base64 string (legacy path)
                image = self._decode_image_direct(image_data)
            elif isinstance(image_data, bytes):
                # Raw binary bytes (optimal path)
                image = self._decode_image_binary(image_data)
            else:
                # numpy array (direct path)
                image = image_data
            
            # Get original dimensions for coordinate scaling
            if hasattr(image, 'size'):
                original_w, original_h = image.size  # PIL Image
            else:
                original_h, original_w = image.shape[:2]  # numpy array
            
            # Direct to YOLO with enhanced parameters
            model = self.model_manager.current_model
            if model is None:
                raise ValueError("No model loaded")
            
            # Build YOLO predict parameters with new high-value features
            predict_params = {
                "source": image,
                "conf": confidence,
                "iou": iou,
                "max_det": max_det,
                "imgsz": settings.input_size,
                "device": self.model_manager.device,
                "verbose": False,
                "save": False,
                # High-Value Parameters
                "classes": detection_classes,  # Official Ultralytics parameter
                "half": half_precision,        # Official Ultralytics FP16
                "augment": test_time_augmentation,  # Official Ultralytics TTA
                "agnostic_nms": class_agnostic_nms  # Official Ultralytics NMS
            }
            
            # Remove None values to use YOLO defaults
            predict_params = {k: v for k, v in predict_params.items() if v is not None}
            
            logger.debug(f"YOLO predict parameters: {predict_params}")
            results = model.predict(**predict_params)
            
            # Parse detections
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # SAINet2.1: Coordinates are already in original image space
                    # since we're not doing manual preprocessing
                    x1 = float(box[0])
                    y1 = float(box[1])
                    x2 = float(box[2])
                    y2 = float(box[3])
                    
                    # Clip to image bounds
                    x1 = max(0, min(x1, original_w))
                    y1 = max(0, min(y1, original_h))
                    x2 = max(0, min(x2, original_w))
                    y2 = max(0, min(y2, original_h))
                    
                    detection = Detection(
                        class_name=DetectionClass.SMOKE if cls == 0 else DetectionClass.FIRE,
                        class_id=cls,
                        confidence=conf,
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    )
                    detections.append(detection)
            
            # Calculate summary statistics
            has_fire = any(d.class_name == DetectionClass.FIRE for d in detections)
            has_smoke = any(d.class_name == DetectionClass.SMOKE for d in detections)
            
            confidence_scores = {}
            for cls in [DetectionClass.FIRE, DetectionClass.SMOKE]:
                class_detections = [d for d in detections if d.class_name == cls]
                if class_detections:
                    confidence_scores[cls] = float(
                        np.mean([d.confidence for d in class_detections])
                    )
                else:
                    confidence_scores[cls] = 0.0
            
            # Generate annotated image if requested
            annotated_image_b64 = None
            if return_annotated:
                try:
                    logger.info(f"Generating annotated image, detections: {len(results[0].boxes) if results[0].boxes is not None else 0}")
                    
                    # Convert PIL to numpy for YOLO plot if needed
                    if hasattr(image, 'size'):  # PIL Image
                        img_array = np.array(image)
                        logger.debug(f"Converted PIL to numpy: {img_array.shape}")
                    else:  # Already numpy
                        img_array = image
                        logger.debug(f"Using numpy array directly: {img_array.shape}")
                    
                    # YOLO plot with configurable annotation parameters
                    logger.debug("Calling YOLO plot...")
                    annotated = results[0].plot(
                        img=img_array,
                        line_width=line_width,      # User-configurable line width
                        font_size=None,             # Auto-calculate optimal font size
                        labels=show_labels,         # User-configurable labels
                        boxes=True,                 # Always show bounding boxes
                        conf=show_confidence        # User-configurable confidence scores
                    )
                    logger.debug(f"YOLO plot successful, output shape: {annotated.shape}, dtype: {annotated.dtype}")
                    
                    # CRITICAL FIX: YOLO plot() returns RGB, but cv2.imencode needs BGR
                    # The issue was that we were double-converting or converting wrong direction
                    # According to ultralytics docs, plot() returns RGB format consistently
                    
                    # YOLO plot returns RGB, convert to BGR for cv2.imencode
                    # This should fix the red/blue color swap issue
                    logger.debug("Converting RGB to BGR...")
                    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    logger.debug(f"BGR conversion successful: {annotated_bgr.shape}")
                    
                    encode_params = [
                        cv2.IMWRITE_JPEG_QUALITY, 98,
                        cv2.IMWRITE_JPEG_OPTIMIZE, 1
                    ]
                    logger.debug("Encoding to JPEG...")
                    success, buffer = cv2.imencode('.jpg', annotated_bgr, encode_params)
                    
                    if success:
                        annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
                        logger.info(f"Annotation successful, base64 length: {len(annotated_image_b64)}")
                    else:
                        logger.error("cv2.imencode failed")
                        
                except Exception as e:
                    logger.error(f"Annotation generation failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            processing_time = (time.time() - start_time) * 1000

            # Determine detection mode and active classes
            from .config import settings as config_settings
            actual_detection_classes = detection_classes or config_settings.default_detection_classes
            if actual_detection_classes == [0]:
                detection_mode = "smoke-only"
                active_classes = ["smoke"]
            elif actual_detection_classes == [1]:
                detection_mode = "fire-only"
                active_classes = ["fire"]
            elif actual_detection_classes == [0, 1] or actual_detection_classes is None:
                detection_mode = "both"
                active_classes = ["smoke", "fire"]
            else:
                detection_mode = "custom"
                active_classes = ["smoke" if 0 in actual_detection_classes else None, "fire" if 1 in actual_detection_classes else None]
                active_classes = [c for c in active_classes if c is not None]

            # Build inference context for enhanced logging
            context = None
            if camera_id:
                from .inference_context import InferenceContext

                smoke_count = sum(1 for d in detections if d.class_name == DetectionClass.SMOKE)
                fire_count = sum(1 for d in detections if d.class_name == DetectionClass.FIRE)
                max_conf = max((d.confidence for d in detections), default=0.0)
                avg_conf = np.mean([d.confidence for d in detections]) if detections else None

                # Extract n8n context from metadata if present
                n8n_workflow_id = metadata.get('n8n_workflow_id') if metadata else None
                n8n_execution_id = metadata.get('n8n_execution_id') if metadata else None
                source = metadata.get('source', 'api-direct') if metadata else 'api-direct'

                context = InferenceContext(
                    request_id=request_id,
                    camera_id=camera_id,
                    detections=detections,
                    detection_count=len(detections),
                    smoke_count=smoke_count,
                    fire_count=fire_count,
                    max_confidence=max_conf,
                    avg_confidence=float(avg_conf) if avg_conf is not None else None,
                    processing_time_ms=processing_time,
                    model_inference_time_ms=None,  # TODO: Track separate model time
                    image_width=original_w,
                    image_height=original_h,
                    model_version=self.model_manager.current_model_name or "unknown",
                    confidence_threshold=confidence,
                    iou_threshold=iou,
                    detection_classes=actual_detection_classes,
                    source=source,
                    n8n_workflow_id=n8n_workflow_id,
                    n8n_execution_id=n8n_execution_id,
                    metadata=metadata or {}
                )

            # Calculate alert level using enhanced alert manager
            from .alert_manager import alert_manager
            alert_level = await alert_manager.determine_alert_level(
                detections=detections,
                camera_id=camera_id,
                context=context
            )

            response = InferenceResponse(
                request_id=request_id,
                timestamp=datetime.utcnow(),
                processing_time_ms=processing_time,
                image_size={"width": original_w, "height": original_h},
                detections=detections,
                detection_count=len(detections),
                has_fire=has_fire,
                has_smoke=has_smoke,
                confidence_scores=confidence_scores,
                alert_level=alert_level,
                detection_mode=detection_mode,
                active_classes=active_classes,
                camera_id=camera_id,  # Echo back for n8n verification
                annotated_image=annotated_image_b64,
                version=self.model_manager.current_model_name or "unknown",
                metadata=metadata or {}
            )
            
            # Cache completely removed to fix identical output bug
            # if cache_key:
            #     self._update_cache(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    async def infer_batch(
        self,
        images: List[Union[str, np.ndarray]],
        request_id: str,
        **kwargs
    ) -> List[InferenceResponse]:
        """Run inference on multiple images"""
        results = []
        
        for idx, image in enumerate(images):
            sub_request_id = f"{request_id}_{idx}"
            result = await self.infer(image, sub_request_id, **kwargs)
            results.append(result)
        
        return results


# Global instance
inference_engine = InferenceEngine()