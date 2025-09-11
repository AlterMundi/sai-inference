# SAI Inference Procedure - Complete Technical Specification

## Model Information
- **Model**: SAINet v2.1 (YOLOv8s-based)
- **File**: `models/last.pt` (116MB)
- **Classes**: 2 classes
  - `0`: smoke
  - `1`: fire
- **Architecture**: YOLOv8s
- **Training**: Specialized for fire/smoke detection with Datalitycs dataset

## Input Specifications

### 1. Image Input Formats

#### **File Upload (Primary)**
```bash
curl -X POST http://localhost:8888/api/v1/infer/file \
  -F "file=@image.jpg" \
  -F "confidence_threshold=0.45"
```

#### **URL Reference**
```json
{
  "image_url": "https://example.com/fire_image.jpg"
}
```

#### **Base64 Encoded (Compatibility)**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."
}
```
- **Format**: JPEG, PNG, BMP, TIFF, WebP
- **Size**: Up to 50MB
- **Color**: RGB (auto-converted if needed)
- **Resolution**: Any (auto-resized to 864x864)

### 2. Processing Parameters

```json
{
  "confidence_threshold": 0.13,    // Min confidence (0.0-1.0), default: 0.45
  "iou_threshold": 0.4,           // NMS IoU threshold (0.0-1.0), default: 0.45
  "max_detections": 100,           // Max detections per image, default: 100
  "return_image": true,           // Return annotated image, default: false
  "metadata": {                    // Optional metadata
    "source": "camera_01",
    "timestamp": "2024-09-06T10:30:00Z"
  }
}
```

## Processing Pipeline

### **YOLO Inference with Automatic Preprocessing**

The SAI Inference Service uses YOLO's built-in preprocessing pipeline for optimal performance and accuracy. No manual preprocessing is performed.

```python
def run_inference(image: PIL.Image | np.ndarray) -> Results:
    """
    Run SAINet2.1 inference with YOLO's automatic preprocessing
    """
    # YOLO automatically handles all preprocessing internally:
    # 1. Image validation and conversion to RGB
    # 2. Letterboxing (resize + pad) to target resolution  
    # 3. Normalization (0-255 → 0.0-1.0)
    # 4. Tensor conversion (HWC → CHW, numpy → torch)
    
    results = model.predict(
        source=image,                 # PIL Image or numpy array input
        conf=confidence_threshold,    # Confidence filtering (0.13)
        iou=iou_threshold,           # NMS IoU threshold (0.4)
        max_det=max_detections,      # Maximum detections (100)
        imgsz=864,                   # Target resolution (864px optimized)
        device=device,               # CPU/CUDA
        verbose=False,               # Silent mode
        save=False                   # No file saving
    )
    return results[0]  # Single image result
```

### **YOLO's Internal Letterboxing Process**

YOLO uses the **LetterBox** algorithm for preprocessing, which:

1. **Maintains aspect ratio** by calculating optimal scale factor
2. **Resizes image** using linear interpolation  
3. **Adds gray padding** (RGB: 114,114,114) to reach target size
4. **Centers the image** within the padded frame
5. **Returns coordinates** in original image space automatically

```python
# Equivalent to YOLO's internal LetterBox (for reference only)
letterbox = LetterBox(
    new_shape=(864, 864),        # Target size from settings.input_size
    auto=False,                  # Fixed size (not minimum rectangle)
    center=True,                 # Center the resized image
    stride=32,                   # Model stride alignment
    padding_value=114,           # Gray padding value
    interpolation=cv2.INTER_LINEAR  # Linear interpolation
)
```

### **Detection Post-processing**

```python
def process_detections(results) -> List[Detection]:
    """
    Process YOLO results into structured detections
    """
    detections = []
    
    if results.boxes is not None:
        for i in range(len(results.boxes)):
            # Extract detection data (coordinates already in original image space)
            box = results.boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
            conf = float(results.boxes.conf[i].cpu().numpy())
            cls = int(results.boxes.cls[i].cpu().numpy())
            
            # Create structured detection
            detection = {
                "class_name": "smoke" if cls == 0 else "fire",
                "class_id": cls,
                "confidence": conf,
                "bbox": {
                    "x1": float(box[0]), "y1": float(box[1]),
                    "x2": float(box[2]), "y2": float(box[3]),
                    "width": float(box[2] - box[0]),
                    "height": float(box[3] - box[1]),
                    "center": [float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)]
                }
            }
            detections.append(detection)
    
    return detections
```

## Output Specifications

### **Primary Response Format**

```json
{
  "request_id": "uuid-string",
  "timestamp": "2024-09-06T10:30:15.123Z",
  "processing_time_ms": 85.6,
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "detections": [
    {
      "class_name": "fire",
      "class_id": 1,
      "confidence": 0.847,
      "bbox": {
        "x1": 450.2,
        "y1": 200.8,
        "x2": 650.5,
        "y2": 400.1,
        "width": 200.3,
        "height": 199.3,
        "center": [550.35, 300.45]
      },
      "metadata": {}
    },
    {
      "class_name": "smoke", 
      "class_id": 0,
      "confidence": 0.623,
      "bbox": {
        "x1": 300.0,
        "y1": 150.0, 
        "x2": 500.0,
        "y2": 250.0,
        "width": 200.0,
        "height": 100.0,
        "center": [400.0, 200.0]
      },
      "metadata": {}
    }
  ],
  "detection_count": 2,
  "has_fire": true,
  "has_smoke": true,
  "confidence_scores": {
    "fire": 0.847,    // Average confidence for fire detections
    "smoke": 0.623    // Average confidence for smoke detections  
  },
  "annotated_image": null,  // Base64 if return_image=true
  "model_version": "sai_v2.1.pt",
  "metadata": {
    "source": "camera_01"
  }
}
```

### **Alert Level Classification**

```python
def determine_alert_level(detections) -> str:
    """Alert level determination logic"""
    fire_count = sum(1 for d in detections if d.class_name == "fire")
    smoke_count = sum(1 for d in detections if d.class_name == "smoke")
    max_confidence = max((d.confidence for d in detections), default=0)
    
    if fire_count > 2 or (fire_count > 0 and max_confidence > 0.8):
        return "critical"  # Multiple fires or high-confidence fire
    elif fire_count > 0:
        return "high"      # Fire detected
    elif smoke_count > 2 or (smoke_count > 0 and max_confidence > 0.7):
        return "medium"    # Multiple smoke or high-confidence smoke
    elif smoke_count > 0:
        return "low"       # Smoke detected
    return "none"          # No detections
```

## Integration Examples

### **n8n HTTP Request Node**
```json
{
  "method": "POST",
  "url": "http://localhost:8888/api/v1/infer",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": {
    "image": "{{$binary.data}}",
    "confidence_threshold": 0.45,
    "return_image": false,
    "metadata": {
      "workflow": "fire_detection",
      "camera": "{{$json.camera_id}}"
    }
  }
}
```

### **Python Client**
```python
import requests
import base64

# Load image
with open("test_image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Send inference request
response = requests.post("http://localhost:8888/api/v1/infer", json={
    "image": image_b64,
    "confidence_threshold": 0.45
})

result = response.json()
print(f"Detections: {result['detection_count']}")
print(f"Has fire: {result['has_fire']}")
```

### **cURL Example**
```bash
curl -X POST http://localhost:8888/api/v1/infer \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$(base64 -w 0 image.jpg)'",
    "confidence_threshold": 0.45,
    "return_image": false
  }'
```

## Error Handling

### **Common Error Responses**
```json
{
  "error": "Invalid image format",
  "detail": "Image must be in JPEG, PNG, BMP, TIFF, or WebP format",
  "timestamp": "2024-09-06T10:30:15.123Z",
  "request_id": "uuid-string"
}
```

### **HTTP Status Codes**
- **200**: Success
- **400**: Invalid request (bad image, parameters)
- **413**: File too large (>50MB)
- **422**: Validation error
- **500**: Internal server error
- **503**: Service unavailable (model not loaded)

The SAINet2.1 model provides state-of-the-art fire and smoke detection with 864px optimized resolution processing for real-time monitoring applications.