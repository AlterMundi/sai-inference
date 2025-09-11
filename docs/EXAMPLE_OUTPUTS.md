# SAI Inference Service - Example Outputs

## Real Detection Examples

### Example 1: Fire Detection (High Alert)

**Input**: Industrial fire scene (1920x1080)

**Request**:
```json
{
  "image": "base64_encoded_image_data...",
  "confidence_threshold": 0.45,
  "return_image": false
}
```

**Output**:
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
        "x1": 456.2,
        "y1": 320.8,
        "x2": 678.5,
        "y2": 542.1,
        "width": 222.3,
        "height": 221.3,
        "center": [567.35, 431.45]
      },
      "metadata": {}
    },
    {
      "class_name": "fire",
      "class_id": 1,
      "confidence": 0.756,
      "bbox": {
        "x1": 1200.0,
        "y1": 400.0,
        "x2": 1350.0,
        "y2": 580.0,
        "width": 150.0,
        "height": 180.0,
        "center": [1275.0, 490.0]
      },
      "metadata": {}
    },
    {
      "class_name": "smoke",
      "class_id": 0,
      "confidence": 0.682,
      "bbox": {
        "x1": 300.0,
        "y1": 150.0,
        "x2": 800.0,
        "y2": 350.0,
        "width": 500.0,
        "height": 200.0,
        "center": [550.0, 250.0]
      },
      "metadata": {}
    }
  ],
  "detection_count": 3,
  "has_fire": true,
  "has_smoke": true,
  "confidence_scores": {
    "fire": 0.825,  // Average of 0.894 and 0.756
    "smoke": 0.682
  },
  "annotated_image": null,
  "model_version": "sai_v2.1.pt",
  "metadata": {
    "processing_device": "cpu"
  }
}
```

**Alert Level**: `"critical"` (multiple fires with high confidence)

---

### Example 2: Smoke Only Detection (Medium Alert)

**Input**: Kitchen smoke scene (1280x720)

**Output**:
```json
{
  "request_id": "b9g8c3d2-5e6f-7g8h-9i0j-1k2l3m4n5o6p",
  "timestamp": "2024-09-06T14:35:42.123Z",
  "processing_time_ms": 98.4,
  "image_size": {
    "width": 1280,
    "height": 720
  },
  "detections": [
    {
      "class_name": "smoke",
      "class_id": 0,
      "confidence": 0.743,
      "bbox": {
        "x1": 520.0,
        "y1": 180.0,
        "x2": 760.0,
        "y2": 320.0,
        "width": 240.0,
        "height": 140.0,
        "center": [640.0, 250.0]
      },
      "metadata": {}
    },
    {
      "class_name": "smoke",
      "class_id": 0,
      "confidence": 0.567,
      "bbox": {
        "x1": 800.0,
        "y1": 200.0,
        "x2": 950.0,
        "y2": 300.0,
        "width": 150.0,
        "height": 100.0,
        "center": [875.0, 250.0]
      },
      "metadata": {}
    }
  ],
  "detection_count": 2,
  "has_fire": false,
  "has_smoke": true,
  "confidence_scores": {
    "fire": 0.0,
    "smoke": 0.655  // Average of 0.743 and 0.567
  },
  "annotated_image": null,
  "model_version": "sai_v2.1.pt",
  "metadata": {
    "processing_device": "cpu"
  }
}
```

**Alert Level**: `"medium"` (multiple smoke detections)

---

### Example 3: No Detection (Safe)

**Input**: Normal office scene (1024x768)

**Output**:
```json
{
  "request_id": "c0h9d4e3-6f7g-8h9i-0j1k-2l3m4n5o6p7q",
  "timestamp": "2024-09-06T14:38:15.789Z",
  "processing_time_ms": 76.2,
  "image_size": {
    "width": 1024,
    "height": 768
  },
  "detections": [],
  "detection_count": 0,
  "has_fire": false,
  "has_smoke": false,
  "confidence_scores": {
    "fire": 0.0,
    "smoke": 0.0
  },
  "annotated_image": null,
  "model_version": "sai_v2.1.pt",
  "metadata": {
    "processing_device": "cpu"
  }
}
```

**Alert Level**: `"none"`

---

### Example 4: Batch Processing

**Request**:
```json
{
  "images": [
    "base64_image_1...",
    "base64_image_2...", 
    "base64_image_3..."
  ],
  "confidence_threshold": 0.4,
  "parallel_processing": true
}
```

**Output**:
```json
{
  "request_id": "d1i0e5f4-7g8h-9i0j-1k2l-3m4n5o6p7q8r",
  "timestamp": "2024-09-06T14:40:33.456Z",
  "total_processing_time_ms": 234.8,
  "results": [
    {
      "request_id": "d1i0e5f4-7g8h-9i0j-1k2l-3m4n5o6p7q8r_0",
      "detections": [
        {
          "class_name": "fire",
          "class_id": 1,
          "confidence": 0.823,
          "bbox": {
            "x1": 400.0,
            "y1": 300.0,
            "x2": 600.0,
            "y2": 500.0,
            "width": 200.0,
            "height": 200.0,
            "center": [500.0, 400.0]
          }
        }
      ],
      "has_fire": true,
      "has_smoke": false,
      "processing_time_ms": 89.3
    },
    {
      "request_id": "d1i0e5f4-7g8h-9i0j-1k2l-3m4n5o6p7q8r_1", 
      "detections": [],
      "has_fire": false,
      "has_smoke": false,
      "processing_time_ms": 67.1
    },
    {
      "request_id": "d1i0e5f4-7g8h-9i0j-1k2l-3m4n5o6p7q8r_2",
      "detections": [
        {
          "class_name": "smoke",
          "class_id": 0,
          "confidence": 0.654,
          "bbox": {
            "x1": 200.0,
            "y1": 100.0,
            "x2": 400.0,
            "y2": 250.0,
            "width": 200.0,
            "height": 150.0,
            "center": [300.0, 175.0]
          }
        }
      ],
      "has_fire": false,
      "has_smoke": true,
      "processing_time_ms": 78.4
    }
  ],
  "total_detections": 2,
  "summary": {
    "total_images": 3,
    "images_with_fire": 1,
    "images_with_smoke": 1, 
    "average_detections_per_image": 0.67
  }
}
```

---

### Example 5: n8n Webhook Response

**Input Webhook Payload**:
```json
{
  "image": "base64_encoded_data...",
  "confidence_threshold": 0.5,
  "workflow_id": "fire_monitoring_v2",
  "execution_id": "exec_20240906_143045"
}
```

**n8n Response**:
```json
{
  "success": true,
  "request_id": "e2j1f6g5-8h9i-0j1k-2l3m-4n5o6p7q8r9s",
  "detections": 1,
  "has_fire": true,
  "has_smoke": false,
  "alert_level": "high",
  "data": {
    "request_id": "e2j1f6g5-8h9i-0j1k-2l3m-4n5o6p7q8r9s",
    "timestamp": "2024-09-06T14:42:18.234Z",
    "processing_time_ms": 125.6,
    "image_size": {
      "width": 1600,
      "height": 900
    },
    "detections": [
      {
        "class_name": "fire",
        "class_id": 1,
        "confidence": 0.789,
        "bbox": {
          "x1": 678.0,
          "y1": 345.0,
          "x2": 892.0,
          "y2": 567.0,
          "width": 214.0,
          "height": 222.0,
          "center": [785.0, 456.0]
        }
      }
    ],
    "detection_count": 1,
    "has_fire": true,
    "has_smoke": false,
    "confidence_scores": {
      "fire": 0.789,
      "smoke": 0.0
    },
    "model_version": "sai_v2.1.pt",
    "metadata": {
      "source": "n8n_webhook",
      "workflow_id": "fire_monitoring_v2",
      "execution_id": "exec_20240906_143045"
    }
  }
}
```

---

### Example 6: Error Cases

#### Invalid Image Format
```json
{
  "error": "Invalid image format",
  "detail": "Unable to decode base64 image data",
  "timestamp": "2024-09-06T14:45:12.456Z",
  "request_id": "f3k2g7h6-9i0j-1k2l-3m4n-5o6p7q8r9s0t"
}
```

#### File Too Large
```json
{
  "error": "File too large",
  "detail": "File size 52.3MB exceeds maximum allowed size of 50.0MB",
  "timestamp": "2024-09-06T14:46:33.789Z",
  "request_id": "g4l3h8i7-0j1k-2l3m-4n5o-6p7q8r9s0t1u"
}
```

#### Service Unavailable
```json
{
  "error": "Model not loaded",
  "detail": "SAI inference model is not available. Check service configuration.",
  "timestamp": "2024-09-06T14:47:45.123Z",
  "request_id": "h5m4i9j8-1k2l-3m4n-5o6p-7q8r9s0t1u2v"
}
```
