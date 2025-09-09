"""
SAI Mosaic Inference Engine
Processes large images by splitting into overlapping 640x640 crops for optimal detection
"""
import numpy as np
from PIL import Image
import time
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
from datetime import datetime
import io
import base64
import cv2

from .config import settings
from .models import (
    Detection, BoundingBox, DetectionClass,
    InferenceResponse, ModelInfo
)
from .inference import inference_engine

logger = logging.getLogger(__name__)


class MosaicInferenceEngine:
    """Mosaic-based inference for large images using 640x640 crops"""
    
    def __init__(self):
        self.crop_size = 640
        self.overlap_pixels = 64  # Overlap between adjacent crops
        
    def calculate_mosaic_grid(self, img_width: int, img_height: int) -> Tuple[List[Tuple[int, int, int, int]], int, int]:
        """
        Calculate optimal crop positions for mosaic inference
        
        Args:
            img_width: Original image width (e.g., 2880)
            img_height: Original image height (e.g., 1616)
            
        Returns:
            List of crop boxes (x1, y1, x2, y2), grid_width, grid_height
        """
        crop_boxes = []
        
        # Calculate horizontal crops (5 wide for 2880px)
        effective_width = self.crop_size - self.overlap_pixels
        cols = max(1, (img_width - self.overlap_pixels) // effective_width)
        if (img_width - self.overlap_pixels) % effective_width > 0:
            cols += 1
        
        # Calculate vertical crops (3 high for 1616px with 304px total overlap)
        effective_height = self.crop_size - self.overlap_pixels
        rows = max(1, (img_height - self.overlap_pixels) // effective_height)
        if (img_height - self.overlap_pixels) % effective_height > 0:
            rows += 1
            
        # Distribute remaining overlap evenly
        total_vertical_overlap = max(0, (rows * self.crop_size) - img_height)
        vertical_overlap_per_crop = total_vertical_overlap // max(1, rows - 1) if rows > 1 else 0
        
        logger.info(f"Mosaic grid: {cols}x{rows} = {cols * rows} crops")
        logger.info(f"Image size: {img_width}x{img_height}, crop size: {self.crop_size}x{self.crop_size}")
        logger.info(f"Horizontal overlap: {self.overlap_pixels}px, vertical overlap: {vertical_overlap_per_crop}px")
        
        for row in range(rows):
            for col in range(cols):
                # Calculate crop position
                x1 = col * effective_width
                y1 = row * (self.crop_size - vertical_overlap_per_crop)
                
                x2 = min(x1 + self.crop_size, img_width)
                y2 = min(y1 + self.crop_size, img_height)
                
                # Adjust if crop is smaller than expected (edge cases)
                if x2 - x1 < self.crop_size and x1 > 0:
                    x1 = max(0, x2 - self.crop_size)
                if y2 - y1 < self.crop_size and y1 > 0:
                    y1 = max(0, y2 - self.crop_size)
                
                crop_boxes.append((x1, y1, x2, y2))
                
        logger.info(f"Generated {len(crop_boxes)} crop boxes")
        return crop_boxes, cols, rows
    
    def extract_crop(self, image: Image.Image, crop_box: Tuple[int, int, int, int]) -> Image.Image:
        """Extract a crop from the original image"""
        x1, y1, x2, y2 = crop_box
        crop = image.crop((x1, y1, x2, y2))
        
        # Ensure crop is exactly 640x640 by padding if necessary
        if crop.size != (self.crop_size, self.crop_size):
            # Create a black canvas and paste the crop
            padded_crop = Image.new('RGB', (self.crop_size, self.crop_size), (0, 0, 0))
            padded_crop.paste(crop, (0, 0))
            crop = padded_crop
            
        return crop
    
    def map_detections_to_original(self, detections: List[Detection], crop_box: Tuple[int, int, int, int]) -> List[Detection]:
        """Map detection coordinates from crop space back to original image space"""
        x_offset, y_offset, _, _ = crop_box
        mapped_detections = []
        
        for detection in detections:
            # Map bounding box coordinates
            mapped_bbox = BoundingBox(
                x1=detection.bbox.x1 + x_offset,
                y1=detection.bbox.y1 + y_offset,
                x2=detection.bbox.x2 + x_offset,
                y2=detection.bbox.y2 + y_offset
            )
            
            # Create new detection with mapped coordinates
            mapped_detection = Detection(
                class_name=detection.class_name,
                class_id=detection.class_id,
                confidence=detection.confidence,
                bbox=mapped_bbox,
                metadata={
                    **detection.metadata,
                    "crop_origin": f"{x_offset},{y_offset}",
                    "crop_box": f"{crop_box[0]},{crop_box[1]},{crop_box[2]},{crop_box[3]}"
                }
            )
            mapped_detections.append(mapped_detection)
            
        return mapped_detections
    
    def remove_duplicate_detections(self, all_detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
        """Remove overlapping detections from mosaic inference using NMS"""
        if len(all_detections) <= 1:
            return all_detections
            
        # Group detections by class
        fire_detections = [d for d in all_detections if d.class_name == DetectionClass.FIRE]
        smoke_detections = [d for d in all_detections if d.class_name == DetectionClass.SMOKE]
        
        def nms_by_class(detections: List[Detection]) -> List[Detection]:
            if len(detections) <= 1:
                return detections
                
            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x.confidence, reverse=True)
            
            keep = []
            while detections:
                # Keep the highest confidence detection
                current = detections.pop(0)
                keep.append(current)
                
                # Remove overlapping detections
                remaining = []
                for det in detections:
                    iou = self.calculate_iou(current.bbox, det.bbox)
                    if iou < iou_threshold:
                        remaining.append(det)
                    else:
                        logger.debug(f"Removed overlapping detection (IoU: {iou:.3f})")
                        
                detections = remaining
                
            return keep
        
        # Apply NMS to each class separately
        final_detections = []
        final_detections.extend(nms_by_class(fire_detections))
        final_detections.extend(nms_by_class(smoke_detections))
        
        logger.info(f"NMS: {len(all_detections)} → {len(final_detections)} detections")
        return final_detections
    
    def calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Calculate intersection area
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
        area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def generate_mosaic_annotated_image(self, original_image: Image.Image, detections: List[Detection]) -> Optional[str]:
        """Generate annotated image with all mosaic detections"""
        if not detections:
            return None
            
        try:
            # Convert PIL to numpy array
            img_array = np.array(original_image)
            
            # Draw detections manually (similar to YOLO plot but for mosaic results)
            for detection in detections:
                bbox = detection.bbox
                color = (0, 255, 0) if detection.class_name == DetectionClass.FIRE else (255, 165, 0)  # Green for fire, orange for smoke
                
                # Draw bounding box
                cv2.rectangle(img_array, 
                            (int(bbox.x1), int(bbox.y1)), 
                            (int(bbox.x2), int(bbox.y2)), 
                            color, 3)
                
                # Draw label with confidence
                label = f"{detection.class_name}: {detection.confidence:.2f}"
                font_scale = max(original_image.size[0] / 1000, 0.5)
                thickness = max(2, int(font_scale * 2))
                
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Draw label background
                cv2.rectangle(img_array,
                            (int(bbox.x1), int(bbox.y1) - text_height - 10),
                            (int(bbox.x1) + text_width, int(bbox.y1)),
                            color, -1)
                
                # Draw label text
                cv2.putText(img_array, label,
                          (int(bbox.x1), int(bbox.y1) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Convert RGB to BGR for cv2.imencode
            annotated_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Encode to JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 98, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            success, buffer = cv2.imencode('.jpg', annotated_bgr, encode_params)
            
            if success:
                return base64.b64encode(buffer).decode('utf-8')
            else:
                logger.error("Failed to encode mosaic annotated image")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate mosaic annotated image: {e}")
            return None
    
    async def infer_mosaic(
        self,
        image_data: Union[str, bytes, np.ndarray],
        request_id: str,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        return_annotated: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> InferenceResponse:
        """Run mosaic inference on large image by processing 640x640 crops"""
        
        start_time = time.time()
        
        # Use defaults if not provided
        confidence = confidence_threshold or settings.confidence_threshold
        iou = iou_threshold or settings.iou_threshold
        max_det = max_detections or settings.max_detections
        
        try:
            # Load original image using existing inference engine logic
            if isinstance(image_data, str):
                image = inference_engine._decode_image_direct(image_data)
            elif isinstance(image_data, bytes):
                image = inference_engine._decode_image_binary(image_data)
            else:
                image = image_data
                
            original_w, original_h = image.size
            logger.info(f"Mosaic inference starting - Original size: {original_w}x{original_h}")
            
            # Calculate mosaic grid
            crop_boxes, grid_cols, grid_rows = self.calculate_mosaic_grid(original_w, original_h)
            
            # Process each crop
            all_detections = []
            crop_results = []
            
            for i, crop_box in enumerate(crop_boxes):
                logger.debug(f"Processing crop {i+1}/{len(crop_boxes)}: {crop_box}")
                
                # Extract crop
                crop_image = self.extract_crop(image, crop_box)
                
                # Convert crop to bytes for inference (reuse existing pipeline)
                crop_buffer = io.BytesIO()
                crop_image.save(crop_buffer, format='JPEG', quality=95)
                crop_bytes = crop_buffer.getvalue()
                
                # Run inference on crop using existing inference engine
                crop_response = await inference_engine.infer(
                    image_data=crop_bytes,
                    request_id=f"{request_id}_crop_{i}",
                    confidence_threshold=confidence,
                    iou_threshold=iou,
                    max_detections=max_det,
                    return_annotated=False,  # We'll generate our own mosaic annotation
                    metadata={
                        **(metadata or {}),
                        "crop_index": i,
                        "crop_box": crop_box,
                        "mosaic_grid": f"{grid_cols}x{grid_rows}",
                        "source": "mosaic_crop"
                    }
                )
                
                crop_results.append(crop_response)
                
                # Map detections back to original coordinates
                if crop_response.detections:
                    mapped_detections = self.map_detections_to_original(crop_response.detections, crop_box)
                    all_detections.extend(mapped_detections)
                    logger.debug(f"Crop {i+1}: {len(crop_response.detections)} detections")
                
            # Remove duplicate detections from overlapping regions
            final_detections = self.remove_duplicate_detections(all_detections, iou_threshold=0.5)
            
            # Calculate summary statistics (BIG OR logic)
            has_fire = any(d.class_name == DetectionClass.FIRE for d in final_detections)
            has_smoke = any(d.class_name == DetectionClass.SMOKE for d in final_detections)
            
            confidence_scores = {}
            for cls in [DetectionClass.FIRE, DetectionClass.SMOKE]:
                class_detections = [d for d in final_detections if d.class_name == cls]
                if class_detections:
                    confidence_scores[cls] = float(np.mean([d.confidence for d in class_detections]))
                else:
                    confidence_scores[cls] = 0.0
            
            # Generate mosaic annotated image if requested
            annotated_image_b64 = None
            if return_annotated:
                annotated_image_b64 = self.generate_mosaic_annotated_image(image, final_detections)
                if annotated_image_b64:
                    logger.info(f"Mosaic annotated image generated with {len(final_detections)} detections")
            
            processing_time = (time.time() - start_time) * 1000
            
            # Create unified response matching original InferenceResponse format
            response = InferenceResponse(
                request_id=request_id,
                timestamp=datetime.utcnow(),
                processing_time_ms=processing_time,
                image_size={"width": original_w, "height": original_h},
                detections=final_detections,
                detection_count=len(final_detections),
                has_fire=has_fire,
                has_smoke=has_smoke,
                confidence_scores=confidence_scores,
                annotated_image=annotated_image_b64,
                version=f"mosaic-{inference_engine.model_manager.current_model_name or 'unknown'}",
                metadata={
                    **(metadata or {}),
                    "mosaic_inference": True,
                    "grid_size": f"{grid_cols}x{grid_rows}",
                    "total_crops": len(crop_boxes),
                    "crops_with_detections": sum(1 for r in crop_results if r.detections),
                    "total_raw_detections": len(all_detections),
                    "final_detections_after_nms": len(final_detections)
                }
            )
            
            logger.info(f"Mosaic inference completed: {len(final_detections)} detections, has_fire={has_fire}, has_smoke={has_smoke}")
            return response
            
        except Exception as e:
            logger.error(f"Mosaic inference failed: {e}")
            raise


# Global mosaic inference engine instance
mosaic_inference_engine = MosaicInferenceEngine()