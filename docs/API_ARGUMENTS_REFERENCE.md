# API Call Arguments Reference - Based on Actual Code

**Source**: Extracted from `src/main.py` and `src/models.py`
**Last Updated**: December 2025

## Overview

SAI Inference Service provides three primary inference endpoints with different input methods and capabilities.

---

## Endpoint 1: `/api/v1/infer` (Primary - Multipart/Form-Data)

**Method**: `POST`
**Content-Type**: `multipart/form-data`
**Best For**: Direct file uploads, n8n workflows, standard HTTP clients

### Required Parameters

| Parameter | Type | Description | Code Location |
|-----------|------|-------------|---------------|
| `file` | UploadFile | Binary image file (jpg, jpeg, png, bmp, tiff, webp) | Line 299 |

### Core Detection Parameters (Optional)

| Parameter | Type | Default | Range | Description | Code Location |
|-----------|------|---------|-------|-------------|---------------|
| `confidence_threshold` | float | 0.39 | 0.0-1.0 | Minimum confidence score to report detection. Production default: 0.39 | `src/config.py:30` |
| `iou_threshold` | float | 0.1 | 0.0-1.0 | IoU threshold for Non-Maximum Suppression (removes overlapping boxes) | `src/config.py:31` |
| `max_detections` | int | 100 | 1-1000 | Maximum number of detections to return per image | Lines 303, 374 |

### Enhanced Alert System (Optional)

| Parameter | Type | Default | Description | Code Location |
|-----------|------|---------|-------------|---------------|
| `camera_id` | str | None | Camera identifier for temporal alert tracking. Enables escalation logic (low→high→critical) | Lines 305, 386 |

**Alert Escalation Logic** (when `camera_id` provided):
- **No camera_id**: Simple confidence-based alerts (none/low/high)
- **With camera_id**: Temporal tracking with persistence
  - 3 high-confidence (≥0.7) detections in 3h → CRITICAL
  - 3 medium-confidence (0.3-0.7) detections in 30m → HIGH
  - Single detection → LOW or HIGH based on confidence

### High-Value YOLO Parameters (Optional)

| Parameter | Type | Format | Default | Description | Code Location |
|-----------|------|--------|---------|-------------|---------------|
| `detection_classes` | str (JSON) | `"[0]"` or `"[1]"` or `"[0,1]"` | `[0]` (smoke-only) | Filter detection classes. `0`=smoke, `1`=fire. Parsed as JSON array | Lines 307, 346-357, 361 |
| `half_precision` | str (bool) | `"true"`/`"false"` | `"false"` | Enable FP16 inference (2x speed, requires compatible GPU) | Lines 308, 362 |
| `test_time_augmentation` | str (bool) | `"true"`/`"false"` | `"false"` | Enable Test-Time Augmentation (better accuracy, 2-3x slower) | Lines 309, 363 |
| `class_agnostic_nms` | str (bool) | `"true"`/`"false"` | `"false"` | Apply NMS across all classes (suppress fire if overlaps with smoke) | Lines 310, 364 |

**String Parsing** (Lines 341-344):
- Accepts: `"true"`, `"1"`, `"yes"`, `"on"` (case-insensitive) → `True`
- Accepts: `"false"`, `"0"`, `"no"`, `"off"`, `None` → `False`

### Annotation Control (Optional)

| Parameter | Type | Format | Default | Description | Code Location |
|-----------|------|--------|---------|-------------|---------------|
| `return_image` | str (bool) | `"true"`/`"false"` | `"false"` | Return base64-encoded annotated image with bounding boxes | Lines 312, 360, 375 |
| `show_labels` | str (bool) | `"true"`/`"false"` | `"true"` | Include class name labels on annotated image | Lines 313, 365, 382 |
| `show_confidence` | str (bool) | `"true"`/`"false"` | `"true"` | Display confidence scores on annotated image | Lines 314, 366, 383 |
| `line_width` | int | - | Auto | Bounding box line thickness (1-10 pixels) | Lines 315, 384 |

### Processing Options (Optional)

| Parameter | Type | Default | Description | Code Location |
|-----------|------|---------|-------------|---------------|
| `webhook_url` | str | None | Send detection results to webhook URL (async, non-blocking) | Lines 317, 395-403 |

### File Validation

**Allowed Extensions** (Lines 324-329):
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

**Max Upload Size** (Lines 332-337):
- Default: 50 MB (52,428,800 bytes)
- Configurable: `SAI_MAX_UPLOAD` environment variable

### Example cURL Call

```bash
curl -X POST http://localhost:8888/api/v1/infer \
  -F "file=@smoke_image.jpg" \
  -F "confidence_threshold=0.13" \
  -F "iou_threshold=0.4" \
  -F "camera_id=camera_001" \
  -F "detection_classes=[0]" \
  -F "return_image=true" \
  -F "show_labels=true" \
  -F "show_confidence=true" \
  -F "webhook_url=https://n8n.example.com/webhook/alerts"
```

---

## Endpoint 2: `/api/v1/infer/base64` (JSON - Legacy/Secondary)

**Method**: `POST`
**Content-Type**: `application/json`
**Best For**: Applications already using base64, JavaScript clients

### Request Body (JSON)

All parameters are optional except one of `image` or `image_url`:

```json
{
  // Image Input (one required)
  "image": "base64_encoded_image_string",
  "image_url": "https://example.com/image.jpg",

  // Core Detection Parameters
  "confidence_threshold": 0.13,
  "iou_threshold": 0.4,
  "max_detections": 100,

  // Enhanced Alert System
  "camera_id": "camera_001",

  // High-Value YOLO Parameters
  "detection_classes": [0],      // Array of integers, not JSON string
  "half_precision": false,        // Boolean, not string
  "test_time_augmentation": false,
  "class_agnostic_nms": false,

  // Annotation Control
  "return_image": false,
  "show_labels": true,
  "show_confidence": true,
  "line_width": 2,

  // Processing Options
  "webhook_url": "https://n8n.example.com/webhook/alerts",
  "metadata": {
    "workflow_id": "123",
    "custom_field": "value"
  }
}
```

### Parameter Details

| Parameter | Type | Description | Code Location |
|-----------|------|-------------|---------------|
| `image` | str | Base64-encoded image (with or without data URI prefix) | Lines 68, 419 |
| `image_url` | str | URL to download image from (fetched via httpx) | Lines 69, 421-423 |
| `confidence_threshold` | float (0.0-1.0) | Same as multipart endpoint | Lines 72, 431 |
| `iou_threshold` | float (0.0-1.0) | Same as multipart endpoint | Lines 73, 432 |
| `max_detections` | int (1-1000) | Same as multipart endpoint | Lines 74, 433 |
| `camera_id` | str | Same as multipart endpoint | Lines 103, 445 |
| `detection_classes` | List[int] | JSON array (not string): `[0]`, `[1]`, `[0,1]` | Lines 77-81, 436 |
| `half_precision` | bool | Boolean (not string): `true`/`false` | Lines 82-85, 437 |
| `test_time_augmentation` | bool | Boolean (not string) | Lines 86-89, 438 |
| `class_agnostic_nms` | bool | Boolean (not string) | Lines 90-93, 439 |
| `return_image` | bool | Boolean (not string) | Lines 96, 434 |
| `show_labels` | bool | Boolean (not string), default: `true` | Lines 97, 441 |
| `show_confidence` | bool | Boolean (not string), default: `true` | Lines 98, 442 |
| `line_width` | int (1-10) | Bounding box line thickness | Lines 99, 443 |
| `webhook_url` | str | Webhook URL for async notification | Lines 102, 459 |
| `metadata` | dict | Custom metadata (passed through to response) | Lines 104, 446-455 |

### Key Differences from `/infer`

1. **Data Types**: Booleans are actual booleans (`true`/`false`), not strings
2. **Array Format**: `detection_classes` is a JSON array `[0]`, not a string `"[0]"`
3. **Image Input**: Base64 string or URL instead of binary file
4. **Metadata**: Can include custom metadata object

### Example cURL Call

```bash
curl -X POST http://localhost:8888/api/v1/infer/base64 \
  -H "Content-Type: application/json" \
  -d '{
    "image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "confidence_threshold": 0.13,
    "camera_id": "camera_001",
    "detection_classes": [0],
    "return_image": false
  }'
```

---

## Endpoint 3: `/api/v1/infer/mosaic` (Large Images)

**Method**: `POST`
**Content-Type**: `multipart/form-data`
**Best For**: High-resolution images (>1920px), panoramic views, satellite imagery

### Parameters

| Parameter | Type | Default | Description | Code Location |
|-----------|------|---------|-------------|---------------|
| `file` | UploadFile | Required | Binary image file (any size) | Line 478 |
| `confidence_threshold` | float | 0.13 | Detection confidence threshold | Lines 479 |
| `iou_threshold` | float | 0.4 | NMS IoU threshold | Lines 480 |
| `camera_id` | str | None | Camera ID for alert tracking | Lines 481 |
| `return_image` | str (bool) | `"false"` | Return annotated mosaic image | Lines 482 |
| `webhook_url` | str | None | Webhook for async notification | Lines 483 |

### Processing Logic

**Mosaic Inference** (Lines 486-522):
1. **Image Split**: Divides large image into 640×640 overlapping crops
2. **Overlap**: 64px overlap between adjacent crops to avoid edge detection loss
3. **Inference**: Runs YOLO on each crop independently
4. **Merge**: Combines detections with NMS to remove duplicates at crop boundaries
5. **Coordinate Transform**: Converts crop-relative coordinates to original image coordinates

### When to Use

- Images larger than 1920px (YOLO native size: 864px)
- Detection quality more important than speed
- Small objects scattered across large scene
- Panoramic camera views

### Example cURL Call

```bash
curl -X POST http://localhost:8888/api/v1/infer/mosaic \
  -F "file=@large_panorama.jpg" \
  -F "confidence_threshold=0.13" \
  -F "camera_id=panoramic_cam_01" \
  -F "return_image=false"
```

---

## Response Format (All Endpoints)

### InferenceResponse Structure

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-09-30T14:30:00.123456",
  "processing_time_ms": 87.5,

  "image_size": {
    "width": 1920,
    "height": 1080
  },

  "detections": [
    {
      "class_name": "smoke",
      "class_id": 0,
      "confidence": 0.87,
      "bbox": {
        "x1": 120.5,
        "y1": 340.2,
        "x2": 450.8,
        "y2": 680.3
      },
      "metadata": {}
    }
  ],

  "detection_count": 1,
  "has_fire": false,
  "has_smoke": true,

  "confidence_scores": {
    "smoke": 0.87
  },

  "alert_level": "low",
  "detection_mode": "smoke-only",
  "active_classes": ["smoke"],

  "annotated_image": "base64_encoded_image_if_requested",
  "version": "1.0.0",
  "metadata": {
    "filename": "test.jpg",
    "source": "binary_upload"
  }
}
```

### Alert Levels (when `camera_id` provided)

| Level | Condition | Description |
|-------|-----------|-------------|
| `"none"` | No detections or confidence <0.3 | No alert |
| `"low"` | Single medium detection (0.3-0.7) | Initial detection, monitoring |
| `"high"` | Single high detection (≥0.7) OR 3+ medium in 30m | Elevated concern |
| `"critical"` | 3+ high detections (≥0.7) in 3h | Immediate action required |

---

## Environment Variable Defaults

These defaults are used when parameters are not provided:

```bash
# Core Detection (from src/config.py lines 30-35)
SAI_CONFIDENCE=0.13           # Production-optimized for wildfire
SAI_IOU_THRESHOLD=0.4
SAI_INPUT_SIZE=864            # SAINet2.1 optimized resolution
SAI_MAX_DETECTIONS=100

# Detection Mode (line 38-42)
SAI_DETECTION_CLASSES=[0]     # Smoke-only for early warning

# File Upload (lines 62-66)
SAI_MAX_UPLOAD=52428800       # 50MB
SAI_ALLOWED_EXTENSIONS=[".jpg",".jpeg",".png",".bmp",".tiff",".webp"]

# Alert System (lines 68-97)
SAI_DATABASE_URL=postgresql://sai_user:password@localhost/sai_dashboard
SAI_WILDFIRE_LOW_THRESHOLD=0.3
SAI_WILDFIRE_HIGH_THRESHOLD=0.7
SAI_PERSISTENCE_COUNT=3
SAI_ESCALATION_MINUTES=30
SAI_ESCALATION_HOURS=3
```

---

## Webhook Payload Format

When `webhook_url` is provided, the service sends:

```json
{
  "event_type": "detection",
  "timestamp": "2025-09-30T14:30:00.123456",
  "source": "sai-inference",
  "alert_level": "high",
  "data": {
    // Full InferenceResponse object
  }
}
```

**Webhook Behavior** (Lines 395-403):
- Async/non-blocking (uses BackgroundTasks)
- Timeout: 10 seconds
- Errors logged but don't fail main request
- Alert level calculated via `determine_alert_level(camera_id)`

---

## Parameter Validation Rules

### String to Boolean Parsing (Lines 341-344)

```python
# Multipart form data accepts these as "true":
"true", "1", "yes", "on"  (case-insensitive)

# Everything else is "false":
"false", "0", "no", "off", None, ""
```

### JSON Array Parsing (Lines 346-357)

```python
# Valid detection_classes formats:
"[0]"      → [0]      # Smoke only
"[1]"      → [1]      # Fire only
"[0,1]"    → [0,1]    # Both
"[0, 1]"   → [0,1]    # Spaces OK

# Invalid formats (raise 400 error):
"0"        # Not an array
"[0,2]"    # 2 is not a valid class
"smoke"    # String not allowed
```

### File Validation (Lines 324-337)

```python
# Extension check (case-insensitive)
allowed = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

# Size check
if len(file_bytes) > 52_428_800:  # 50 MB
    raise HTTPException(413)
```

---

## Common Usage Patterns

### Pattern 1: Smoke-Only Wildfire Detection
```bash
curl -X POST http://localhost:8888/api/v1/infer \
  -F "file=@camera_frame.jpg" \
  -F "camera_id=tower_cam_03" \
  -F "detection_classes=[0]" \
  -F "confidence_threshold=0.13"
```

### Pattern 2: Fire and Smoke with Alerts
```bash
curl -X POST http://localhost:8888/api/v1/infer \
  -F "file=@image.jpg" \
  -F "camera_id=cam_001" \
  -F "detection_classes=[0,1]" \
  -F "webhook_url=https://alerts.example.com/fire"
```

### Pattern 3: Annotated Image for Review
```bash
curl -X POST http://localhost:8888/api/v1/infer \
  -F "file=@test.jpg" \
  -F "return_image=true" \
  -F "show_labels=true" \
  -F "show_confidence=true" \
  -F "line_width=3"
```

### Pattern 4: High-Resolution Panoramic
```bash
curl -X POST http://localhost:8888/api/v1/infer/mosaic \
  -F "file=@panorama_8k.jpg" \
  -F "camera_id=panoramic_001"
```

---

## Source Code References

All information extracted from:
- **Main API**: `/opt/sai-inference/src/main.py` lines 297-585
- **Models**: `/opt/sai-inference/src/models.py` lines 65-105
- **Config**: `/opt/sai-inference/src/config.py` lines 25-97

**Verification Date**: September 30, 2025
**Codebase Version**: Production installation at `/opt/sai-inference`
