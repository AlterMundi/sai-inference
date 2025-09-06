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
import hashlib
import json

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
        device = settings.model_device.lower()
        
        if device.startswith("cuda"):
            if torch.cuda.is_available():
                if ":" in device:
                    return device
                return "cuda:0"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
        return "cpu"
    
    def load_model(self, model_name: str, model_path: Optional[Path] = None) -> bool:
        """Load a YOLO model"""
        try:
            if model_path is None:
                model_path = settings.model_dir / model_name
            
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
                input_size=settings.input_size,  # 1920px from reference
                confidence_threshold=settings.model_confidence,  # 0.15 from reference
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
        self.cache: Dict[str, InferenceResponse] = {}
        self.cache_timestamps: Dict[str, float] = {}
        
        # Load default model
        self._load_default_model()
    
    def _load_default_model(self):
        """Load the default model on startup"""
        model_path = settings.model_dir / settings.default_model
        if not model_path.exists():
            # Try to copy from SAINet development
            logger.warning(f"Default model not found at {model_path}")
            self._copy_development_model()
        
        self.model_manager.load_model(settings.default_model)
    
    def _copy_development_model(self):
        """Copy model from development directory"""
        try:
            import shutil
            settings.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Try SAINet2.1 first (newest, best performance)
            source = Path("/mnt/n8n-data/SAINet_v1.0/datasets/D-Fire/SAINet2.1/best.pt")
            if source.exists():
                dest = settings.model_dir / "sai_v2.1.pt"
                shutil.copy2(source, dest)
                logger.info(f"Copied SAINet2.1 model to {dest}")
                return
            
            # Fallback to stage2 model
            source = Path("/mnt/n8n-data/SAINet_v1.0/run_stage2/weights/best.pt")
            if source.exists():
                dest = settings.model_dir / "sai_stage2.pt"
                shutil.copy2(source, dest)
                logger.info(f"Copied stage2 model to {dest}")
                
        except Exception as e:
            logger.error(f"Failed to copy development model: {e}")
    
    def _decode_image(self, image_data: str) -> np.ndarray:
        """Decode base64 image to numpy array"""
        # Remove data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return np.array(image)
    
    def _preprocess_image(self, image: np.ndarray) -> tuple:
        """Preprocess image for SAINet2.1 resolution (1920px - reference implementation)"""
        # SAINet2.1 uses 1920px resolution (from reference: imgsz=1920)
        # Let YOLO handle preprocessing internally for best results
        h, w = image.shape[:2]
        target_size = settings.input_size  # 1920 from reference
        
        # For 1920px, we maintain aspect ratio and let YOLO do the final processing
        # This matches the reference implementation behavior
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize to target while maintaining aspect ratio
        if scale != 1.0:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image.copy()
            
        return resized, scale, (0, 0)  # No padding needed, YOLO handles it
    
    def _generate_cache_key(self, image_data: str, params: Dict[str, Any]) -> str:
        """Generate cache key for inference request"""
        key_data = f"{image_data[:100]}{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[InferenceResponse]:
        """Check if result is in cache"""
        if not settings.cache_enabled:
            return None
        
        if cache_key in self.cache:
            timestamp = self.cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp < settings.cache_ttl:
                logger.debug(f"Cache hit for key: {cache_key}")
                return self.cache[cache_key]
            else:
                # Expired
                del self.cache[cache_key]
                del self.cache_timestamps[cache_key]
        
        return None
    
    def _update_cache(self, cache_key: str, response: InferenceResponse):
        """Update cache with new result"""
        if settings.cache_enabled:
            self.cache[cache_key] = response
            self.cache_timestamps[cache_key] = time.time()
            
            # Cleanup old entries if cache is too large
            if len(self.cache) > 1000:
                # Remove oldest 100 entries
                sorted_keys = sorted(
                    self.cache_timestamps.keys(),
                    key=lambda k: self.cache_timestamps[k]
                )
                for key in sorted_keys[:100]:
                    del self.cache[key]
                    del self.cache_timestamps[key]
    
    async def infer(
        self,
        image_data: Union[str, np.ndarray],
        request_id: str,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        return_annotated: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> InferenceResponse:
        """Run inference on an image"""
        
        start_time = time.time()
        
        # Use defaults if not provided
        confidence = confidence_threshold or settings.model_confidence
        iou = iou_threshold or settings.model_iou_threshold
        max_det = max_detections or settings.max_detections
        
        # Check cache
        cache_params = {
            "conf": confidence,
            "iou": iou,
            "max_det": max_det,
            "model": self.model_manager.current_model_name
        }
        
        if isinstance(image_data, str):
            cache_key = self._generate_cache_key(image_data, cache_params)
            cached_result = self._check_cache(cache_key)
            if cached_result:
                cached_result.metadata["cache_hit"] = True
                return cached_result
        else:
            cache_key = None
        
        try:
            # Decode image if base64
            if isinstance(image_data, str):
                image = self._decode_image(image_data)
            else:
                image = image_data
            
            original_h, original_w = image.shape[:2]
            
            # SAINet2.1 Reference Implementation - simplified preprocessing
            # Use original image, let YOLO handle resizing to imgsz=1920
            processed_image = image  # Use original image like reference
            scale = 1.0  # No manual scaling
            padding = (0, 0)  # No manual padding
            
            # Run inference with SAINet2.1 reference parameters
            model = self.model_manager.current_model
            if model is None:
                raise ValueError("No model loaded")
            
            results = model.predict(
                processed_image,
                conf=confidence,  # Default 0.15 from reference
                iou=iou,
                max_det=max_det,
                imgsz=settings.input_size,  # 1920 from reference
                device=self.model_manager.device,
                verbose=False,
                save=False  # Don't save like reference (save=True)
            )
            
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
            if return_annotated and len(results) > 0:
                annotated = results[0].plot()
                _, buffer = cv2.imencode('.jpg', annotated)
                annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            processing_time = (time.time() - start_time) * 1000
            
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
                annotated_image=annotated_image_b64,
                model_version=self.model_manager.current_model_name or "unknown",
                metadata=metadata or {}
            )
            
            # Update cache
            if cache_key:
                self._update_cache(cache_key, response)
            
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