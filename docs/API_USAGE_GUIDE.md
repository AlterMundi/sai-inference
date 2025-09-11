# SAI Inference Service - Complete API Usage Guide

## Service Setup & Access

### 1. Start the Service
```bash
cd ~/REPOS/sai-inference
./setup.sh                    # Install dependencies & copy model
python run.py                 # Start service on port 8888
```

### 2. Access Points
- **API Base**: `http://localhost:8888`
- **Health Check**: `http://localhost:8888/api/v1/health`

## Endpoint Usage

### **Primary File Upload Endpoint** ⭐ **Recommended**

**URL**: `POST /api/v1/infer`
**Content-Type**: `multipart/form-data`

#### **File Upload (Binary)**
```bash
curl -X POST http://localhost:8888/api/v1/infer \
  -F "file=@image.jpg" \
  -F "confidence_threshold=0.13" \
  -F "iou_threshold=0.4" \
  -F "return_image=true"
```

#### **Advanced Parameters (New Features)**
```bash
# Fire-only detection with GPU acceleration
curl -X POST http://localhost:8888/api/v1/infer \
  -F "file=@image.jpg" \
  -F "detection_classes=[1]" \
  -F "half_precision=true" \
  -F "confidence_threshold=0.13"

# Maximum accuracy mode (slower)
curl -X POST http://localhost:8888/api/v1/infer \
  -F "file=@image.jpg" \
  -F "test_time_augmentation=true" \
  -F "class_agnostic_nms=true"
```

### **JSON Base64 Endpoint** (Alternative)

**URL**: `POST /api/v1/infer/base64`
**Content-Type**: `application/json`

#### **Base64 Image Data**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA...",
  "confidence_threshold": 0.13,
  "iou_threshold": 0.4,
  "detection_classes": [0, 1],
  "half_precision": false,
  "test_time_augmentation": false,
  "class_agnostic_nms": false,
  "return_image": true,
  "metadata": {
    "camera_id": "cam_01",
    "location": "warehouse_A"
  }
}
```

#### **Image URL Processing**
```json
{
  "image_url": "https://example.com/fire_scene.jpg",
  "confidence_threshold": 0.13,
  "detection_classes": [1],
  "half_precision": true
}
```

#### **Complete Output Response**

```json
{
  "request_id": "a8f7b2c1-4d5e-6f7g-8h9i-0j1k2l3m4n5o",
  "timestamp": "2024-09-06T14:32:18.456Z",
  "processing_time_ms": 142.7,
  
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  
  "detections": [
    {
      "class_name": "fire",
      "class_id": 1,
      "confidence": 0.894,
      "bbox": {
        "x1": 456.2,    // Top-left X
        "y1": 320.8,    // Top-left Y  
        "x2": 678.5,    // Bottom-right X
        "y2": 542.1,    // Bottom-right Y
        "width": 222.3,
        "height": 221.3,
        "center": [567.35, 431.45]
      },
      "metadata": {}
    }
  ],
  
  "detection_count": 1,
  "has_fire": true,
  "has_smoke": false,
  
  "confidence_scores": {
    "fire": 0.894,    // Average confidence for fire detections
    "smoke": 0.0      // Average confidence for smoke detections
  },
  
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",  // ⭐ ANNOTATED IMAGE
  "model_version": "sai_v2.1.pt",
  
  "metadata": {
    "camera_id": "cam_01",
    "location": "warehouse_A"
  }
}
```

## Advanced Parameters (High-Value Features)

### **Class Filtering (`detection_classes`)**
**Type**: `Array[int]` | **Default**: `null` (all classes)
**Source**: Official Ultralytics YOLO parameter

```json
{
  "detection_classes": [0],     // Smoke-only detection
  "detection_classes": [1],     // Fire-only detection  
  "detection_classes": [0, 1]   // Both classes (default)
}
```

**Benefits**:
- **Reduced False Positives**: Fire-only mode eliminates smoke misclassifications
- **Targeted Alerts**: Smoke-only detection for early warning systems
- **Performance**: Single-class detection reduces computational overhead

### **Half Precision (`half_precision`)**
**Type**: `boolean` | **Default**: `false`
**Source**: Official Ultralytics YOLO FP16 optimization

```json
{
  "half_precision": true   // Enable FP16 inference
}
```

**Benefits**:
- **2x Speed Boost**: Faster inference on compatible NVIDIA GPUs
- **50% Memory Reduction**: Lower VRAM usage
- **Minimal Accuracy Impact**: <1% mAP difference
- **Requirements**: NVIDIA GPU with Tensor Core support (RTX series)

### **Test-Time Augmentation (`test_time_augmentation`)**
**Type**: `boolean` | **Default**: `false`
**Source**: Official Ultralytics YOLO TTA feature

```json
{
  "test_time_augmentation": true   // Enable TTA for maximum accuracy
}
```

**Benefits**:
- **Enhanced Accuracy**: 5-10% improvement in challenging conditions
- **Better Edge Cases**: Improved detection of faint smoke, distant fire
- **Critical Applications**: Justified for life-safety systems
- **Trade-off**: 2-3x slower inference time

### **Class-Agnostic NMS (`class_agnostic_nms`)**
**Type**: `boolean` | **Default**: `false`
**Source**: Official Ultralytics YOLO NMS configuration

```json
{
  "class_agnostic_nms": true   // Suppress overlapping detections across classes
}
```

**Benefits**:
- **Cleaner Output**: Single detection per spatial region
- **Fire/Smoke Overlap**: Better handling of coexistent fire and smoke
- **Simplified Logic**: Easier alert processing for overlapping phenomena

### **Performance Parameter Matrix**

| Parameter | Speed Impact | Accuracy Impact | Use Case |
|-----------|-------------|-----------------|----------|
| `detection_classes=[1]` | +20% faster | Targeted | Fire-only alerts |
| `half_precision=true` | +100% faster | -1% | Real-time processing |
| `test_time_augmentation=true` | -200% slower | +5-10% | Critical analysis |
| `class_agnostic_nms=true` | Neutral | Cleaner output | Overlapping scenarios |

## Annotated Image Feature

### **How Annotated Images Work**

When `"return_image": true`, the API:

1. **Runs inference** on the original image
2. **Draws bounding boxes** around detections
3. **Adds labels** with class name and confidence
4. **Encodes to base64** JPEG format
5. **Returns in response** as `annotated_image` field

### **Annotated Image Output**

The `annotated_image` field contains:
```json
{
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA..."
}
```

This is a **complete data URI** that can be:
- **Displayed directly** in HTML: `<img src="data:image/jpeg;base64,..." />`
- **Saved to file** by decoding the base64
- **Sent to n8n workflows** for further processing
- **Embedded in notifications** or reports

### **Visual Annotations Include**

- **Bounding boxes**: Red for fire, Blue for smoke
- **Confidence labels**: "fire (0.89)" or "smoke (0.67)"
- **Detection count**: Total detections shown
- **Original image preserved**: No cropping or resizing

## Complete API Examples

### **Example 1: Python Client with Annotated Image**

```python
import requests
import base64
from PIL import Image
import io

# Load and encode image
with open("fire_scene.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode('utf-8')

# Send inference request
response = requests.post("http://localhost:8888/api/v1/infer", json={
    "image": image_b64,
    "confidence_threshold": 0.15,
    "return_image": True,  # ⭐ Request annotated image
    "metadata": {
        "source": "security_camera",
        "timestamp": "2024-09-06T14:30:00Z"
    }
})

result = response.json()

# Process results
print(f"🔥 Detections: {result['detection_count']}")
print(f"🔥 Has Fire: {result['has_fire']}")
print(f"💨 Has Smoke: {result['has_smoke']}")

# Save annotated image
if result.get('annotated_image'):
    # Remove data URI prefix
    image_data = result['annotated_image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    
    # Save to file
    with open("annotated_result.jpg", "wb") as f:
        f.write(image_bytes)
    
    print("📸 Annotated image saved as 'annotated_result.jpg'")

# Print detection details
for i, detection in enumerate(result['detections']):
    print(f"Detection {i+1}:")
    print(f"  Class: {detection['class_name']}")
    print(f"  Confidence: {detection['confidence']:.3f}")
    print(f"  Location: ({detection['bbox']['x1']:.1f}, {detection['bbox']['y1']:.1f}) - ({detection['bbox']['x2']:.1f}, {detection['bbox']['y2']:.1f})")
```

### **Example 2: cURL with File Upload**

```bash
# Upload file directly (with annotations)
curl -X POST http://localhost:8888/api/v1/infer/file \
  -F "file=@fire_scene.jpg" \
  -F "confidence_threshold=0.15" \
  -F "return_image=true" \
  --output response.json

# Extract annotated image from response
python3 -c "
import json
import base64

with open('response.json', 'r') as f:
    data = json.load(f)

if 'annotated_image' in data:
    image_data = data['annotated_image'].split(',')[1]
    with open('annotated_output.jpg', 'wb') as f:
        f.write(base64.b64decode(image_data))
    print('Annotated image saved!')
"
```

### **Example 3: n8n Workflow Integration**

#### **n8n HTTP Request Node Configuration**
```json
{
  "method": "POST",
  "url": "http://localhost:8888/api/v1/infer",
  "sendBody": true,
  "bodyContentType": "json",
  "jsonBody": {
    "image": "={{$binary.data}}",
    "confidence_threshold": 0.15,
    "return_image": true,
    "metadata": {
      "workflow": "fire_monitoring",
      "camera": "{{$json.camera_id}}",
      "timestamp": "={{$now}}"
    }
  }
}
```

#### **n8n Response Processing**
```javascript
// In n8n Function node
const result = $input.first().json;

// Check for fire detection
if (result.has_fire) {
    // Send alert with annotated image
    return {
        alert_level: "CRITICAL",
        message: `🔥 FIRE DETECTED! Confidence: ${result.confidence_scores.fire}`,
        detections: result.detection_count,
        annotated_image: result.annotated_image,  // Ready for email/Slack
        location: result.metadata.camera
    };
}

// Log normal status
return {
    status: "normal",
    processing_time: result.processing_time_ms,
    detections: result.detection_count
};
```


**Response**:
```json
{
  "success": true,
  "request_id": "uuid-string",
  "detections": 2,
  "has_fire": true,
  "has_smoke": true,
  "alert_level": "critical",
  "data": {
    "annotated_image": "data:image/jpeg;base64,/9j/...",
    // ... full inference response
  }
}
```

## Batch Processing

### **Multiple Images with Annotations**

```json
{
  "images": [
    "base64_image_1...",
    "base64_image_2...",
    "base64_image_3..."
  ],
  "confidence_threshold": 0.15,
  "return_images": true,  // ⭐ Batch annotation
  "parallel_processing": true
}
```

**Response**:
```json
{
  "request_id": "batch-uuid",
  "total_processing_time_ms": 456.7,
  "results": [
    {
      "request_id": "batch-uuid_0",
      "detections": [/*...*/],
      "annotated_image": "data:image/jpeg;base64,/9j/...",  // Image 1 annotated
      "has_fire": true
    },
    {
      "request_id": "batch-uuid_1", 
      "detections": [],
      "annotated_image": "data:image/jpeg;base64,/9j/...",  // Image 2 annotated
      "has_fire": false
    }
  ],
  "summary": {
    "total_images": 3,
    "images_with_fire": 1,
    "images_with_smoke": 0
  }
}
```

## Integration Patterns

### **Pattern 1: Real-time Monitoring with Annotations**
```python
# Stream processing with visual feedback
while True:
    image = capture_from_camera()
    result = api_request(image, return_image=True)
    
    if result['has_fire']:
        send_alert_with_image(result['annotated_image'])
        
    display_live_feed(result['annotated_image'])
```

### **Pattern 2: Batch Analysis with Reports**
```python
# Process folder of images with visual report
images = load_image_folder()
batch_result = batch_inference(images, return_images=True)

# Generate HTML report with annotated images
html_report = generate_report(batch_result)
save_report("fire_detection_report.html", html_report)
```

### **Pattern 3: n8n Alert System**
```
Image Input → SAI Inference → Fire Detected? → Email Alert (with annotated image)
                           → No Fire → Log to Database
```

## Performance Considerations

### **With Annotations (`return_image=true`)**
- **Processing Time**: +10-20ms (annotation overhead)
- **Response Size**: 2-5x larger (base64 image data)
- **Memory Usage**: +50-100MB (image processing)
- **Network**: Slower transfer due to large response

### **Without Annotations (`return_image=false`)**  
- **Processing Time**: 125-180ms (inference only)
- **Response Size**: ~2-5KB (JSON only)
- **Memory Usage**: Minimal overhead
- **Network**: Fast transfer

### **Recommendations**
- **Development/Debug**: Use `return_image=true` for visual verification
- **Production/High Volume**: Use `return_image=false` for speed
- **Alerts/Reports**: Use `return_image=true` selectively when fire detected
- **Real-time Monitoring**: Toggle based on detection results

The annotated image feature provides powerful visual feedback for debugging, alerts, and reporting while maintaining high-performance inference for production use cases.