# SAI Inference Procedure - Complete Technical Specification

## Model Information
- **Model**: SAINet v2.1 (YOLOv8s-based)
- **File**: `/mnt/n8n-data/SAINet_v1.0/datasets/D-Fire/SAINet2.1/best.pt` (116MB)
- **Classes**: 2 classes
  - `0`: smoke
  - `1`: fire
- **Architecture**: YOLOv8s with SACRED resolution optimization
- **Training**: Specialized for fire/smoke detection with Datalitycs dataset

## Input Specifications

### 1. Image Input Formats

#### **Base64 Encoded (Primary)**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."
}
```
- **Format**: JPEG, PNG, BMP, TIFF, WebP
- **Size**: Up to 50MB
- **Color**: RGB (auto-converted if needed)
- **Resolution**: Any (auto-resized to 896x896)

#### **File Upload**
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

### 2. Processing Parameters

```json
{
  "confidence_threshold": 0.45,    // Min confidence (0.0-1.0), default: 0.45
  "iou_threshold": 0.45,           // NMS IoU threshold (0.0-1.0), default: 0.45
  "max_detections": 100,           // Max detections per image, default: 100
  "return_image": false,           // Return annotated image, default: false
  "metadata": {                    // Optional metadata
    "source": "camera_01",
    "timestamp": "2024-09-06T10:30:00Z"
  }
}
```

## Processing Pipeline

### **Step 1: Image Preprocessing**

```python
def preprocess_image(image: np.ndarray) -> tuple:
    """
    Convert input image to SACRED resolution (896x896)
    """
    h, w = image.shape[:2]
    target_size = 896  # SACRED resolution
    
    # 1. Calculate scaling factor (maintain aspect ratio)
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 2. Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 3. Pad to square (896x896) with gray borders
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)  # Gray padding
    )
    
    return padded, scale, (left, top)
```

### **Step 2: YOLO Inference**

```python
def run_inference(processed_image: np.ndarray) -> Results:
    """
    Run SAINet2.1 inference on preprocessed image
    """
    results = model.predict(
        processed_image,
        conf=confidence_threshold,     # Confidence filtering
        iou=iou_threshold,            # NMS IoU threshold  
        max_det=max_detections,       # Maximum detections
        device=device,                # CPU/CUDA
        verbose=False,                # Silent mode
        imgsz=896                     # SACRED resolution
    )
    return results[0]  # Single image result
```

### **Step 3: Post-processing**

```python
def postprocess_detections(results, scale, padding, original_size):
    """
    Convert YOLO outputs to final detection format
    """
    detections = []
    
    if results.boxes is not None:
        for i, box in enumerate(results.boxes):
            # Extract raw detection data
            xyxy = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
            conf = float(box.conf[0].cpu().numpy())  # Confidence score
            cls = int(box.cls[0].cpu().numpy())  # Class ID (0=smoke, 1=fire)
            
            # Convert coordinates back to original image space
            x1 = (xyxy[0] - padding[0]) / scale
            y1 = (xyxy[1] - padding[1]) / scale  
            x2 = (xyxy[2] - padding[0]) / scale
            y2 = (xyxy[3] - padding[1]) / scale
            
            # Clip to original image bounds
            x1 = max(0, min(x1, original_size[0]))
            y1 = max(0, min(y1, original_size[1]))
            x2 = max(0, min(x2, original_size[0]))
            y2 = max(0, min(y2, original_size[1]))
            
            detection = {
                "class_name": "smoke" if cls == 0 else "fire",
                "class_id": cls,
                "confidence": conf,
                "bbox": {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "center": [(x1 + x2) / 2, (y1 + y2) / 2]
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
    "cache_hit": false,
    "source": "camera_01"
  }
}
```

### **n8n Webhook Response**

```json
{
  "success": true,
  "request_id": "uuid-string", 
  "detections": 2,
  "has_fire": true,
  "has_smoke": true,
  "alert_level": "critical",  // critical/high/medium/low/none
  "data": {
    // Full InferenceResponse object above
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

## Performance Metrics

### **Processing Times** (896x896 SACRED)
- **CPU (Intel Xeon)**: ~150-250ms per image
- **GPU (RTX 3090)**: ~15-30ms per image  
- **GPU (A100)**: ~10-20ms per image

### **Memory Usage**
- **Model Size**: 116MB (SAINet2.1)
- **Runtime Memory**: ~2GB with model loaded
- **Peak Memory**: ~3GB during batch processing

### **Accuracy Metrics** (from training)
- **mAP@0.5**: ~0.85 (fire/smoke combined)
- **Precision**: ~0.82
- **Recall**: ~0.78
- **F1-Score**: ~0.80

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

The SAINet2.1 model provides state-of-the-art fire and smoke detection with optimized SACRED resolution processing for real-time monitoring applications.