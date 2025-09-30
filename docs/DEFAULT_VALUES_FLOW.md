# Default Values Flow - Code Validation

**Question**: Where do default values for API arguments come from?

**Answer**: It's a **three-tier cascade system**, not just `.env` inheritance.

---

## The Three-Tier Default System

### Tier 1: API Endpoint (main.py)
**Location**: `/opt/sai-inference/src/main.py` lines 301-315

```python
@app.post(f"{settings.api_prefix}/infer")
async def infer(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Form(None),  # ← Tier 1: None
    iou_threshold: Optional[float] = Form(None),         # ← Tier 1: None
    max_detections: Optional[int] = Form(None),          # ← Tier 1: None
    detection_classes: Optional[str] = Form(None),       # ← Tier 1: None
    half_precision: Optional[str] = Form("false"),       # ← Tier 1: "false"
    return_image: Optional[str] = Form("false"),         # ← Tier 1: "false"
    show_labels: Optional[str] = Form("true"),           # ← Tier 1: "true"
    ...
):
```

**What happens here**:
- Most parameters default to `None` (not provided)
- Some boolean strings have hardcoded defaults (`"false"`, `"true"`)
- Parameters are passed **as-is** to the inference engine

**Code Reference**: Lines 369-392 in `main.py`
```python
response = await inference_engine.infer(
    image_data=contents,
    confidence_threshold=confidence_threshold,  # ← Still None if not provided
    iou_threshold=iou_threshold,                 # ← Still None if not provided
    max_detections=max_detections,               # ← Still None if not provided
    ...
)
```

---

### Tier 2: Inference Engine (inference.py)
**Location**: `/opt/sai-inference/src/inference.py` lines 356-360

```python
async def infer(
    self,
    confidence_threshold: Optional[float] = None,  # ← Receives None from API
    iou_threshold: Optional[float] = None,
    max_detections: Optional[int] = None,
    detection_classes: Optional[List[int]] = None,
    ...
):
    # THIS IS WHERE DEFAULTS ARE APPLIED
    confidence = confidence_threshold or settings.confidence_threshold  # ← Tier 2
    iou = iou_threshold or settings.iou_threshold                      # ← Tier 2
    max_det = max_detections or settings.max_detections                # ← Tier 2
    detection_classes = detection_classes or settings.default_detection_classes  # ← Tier 2
```

**What happens here**:
- `or` operator: if parameter is `None`, use `settings.*`
- **This is where `.env` values are actually used**
- Applied just before passing to YOLO model

---

### Tier 3: Settings/Environment (config.py)
**Location**: `/opt/sai-inference/src/config.py` lines 25-42

```python
class Settings(BaseSettings):
    # Model Configuration
    confidence_threshold: float = Field(default=0.13, alias="SAI_CONFIDENCE_THRESHOLD")
    iou_threshold: float = Field(default=0.4, alias="SAI_IOU_THRESHOLD")
    input_size: Union[int, Tuple[int, int]] = Field(default=864, alias="SAI_INPUT_SIZE")
    max_detections: int = Field(default=100, alias="SAI_MAX_DETECTIONS")

    default_detection_classes: Optional[List[int]] = Field(
        default=[0],
        alias="SAI_DETECTION_CLASSES",
        description="Filter to specific class IDs (0=smoke, 1=fire)"
    )
```

**What happens here**:
1. **If `.env` file exists**: Read `SAI_CONFIDENCE`, `SAI_IOU_THRESHOLD`, etc.
2. **If `.env` missing or variable not set**: Use hardcoded `default=` value
3. Pydantic BaseSettings handles the loading automatically

---

## Complete Flow Example

### Example 1: User provides value

```bash
curl -X POST http://localhost:8888/api/v1/infer \
  -F "file=@test.jpg" \
  -F "confidence_threshold=0.25"  # ← User provides
```

**Flow**:
1. **API (main.py:301)**: `confidence_threshold = 0.25` (from Form data)
2. **Inference (inference.py:357)**: `confidence = 0.25 or settings.confidence_threshold` → Uses `0.25`
3. **YOLO**: Runs with `conf=0.25`

---

### Example 2: User provides nothing

```bash
curl -X POST http://localhost:8888/api/v1/infer \
  -F "file=@test.jpg"
  # No confidence_threshold provided
```

**Flow**:
1. **API (main.py:301)**: `confidence_threshold = None` (default from `Form(None)`)
2. **Inference (inference.py:357)**: `confidence = None or settings.confidence_threshold`
3. **Settings (config.py:30)**:
   - Checks `.env` for `SAI_CONFIDENCE`
   - If found: Uses `.env` value
   - If not found: Uses `default=0.13`
4. **YOLO**: Runs with `conf=0.13` (or `.env` value)

---

### Example 3: `.env` file has custom value

**.env file**:
```bash
SAI_CONFIDENCE_THRESHOLD=0.20
SAI_IOU_THRESHOLD=0.5
```

**API call**:
```bash
curl -X POST http://localhost:8888/api/v1/infer \
  -F "file=@test.jpg"
```

**Flow**:
1. **Startup**: Pydantic loads `.env`, `settings.confidence_threshold = 0.20`
2. **API**: `confidence_threshold = None`
3. **Inference**: `confidence = None or 0.20` → Uses `0.20`
4. **YOLO**: Runs with `conf=0.20`

---

## The Actual Default Values

### From Hardcoded Defaults (config.py)

| Parameter | Hardcoded Default | .env Variable | Code Line |
|-----------|-------------------|---------------|-----------|
| `confidence_threshold` | 0.13 | `SAI_CONFIDENCE` | config.py:30 |
| `iou_threshold` | 0.4 | `SAI_IOU_THRESHOLD` | config.py:31 |
| `input_size` | 864 | `SAI_INPUT_SIZE` | config.py:34 |
| `max_detections` | 100 | `SAI_MAX_DETECTIONS` | config.py:35 |
| `default_detection_classes` | `[0]` | `SAI_DETECTION_CLASSES` | config.py:38-42 |
| `device` | `"cpu"` | `SAI_DEVICE` | config.py:28 |
| `max_upload_size` | 52428800 (50MB) | `SAI_MAX_UPLOAD` | config.py:62 |
| `allowed_extensions` | `[".jpg", ".jpeg", ...]` | `SAI_ALLOWED_EXTENSIONS` | config.py:63-66 |

### From API Endpoint Defaults (main.py)

| Parameter | API Default | Override Logic | Code Line |
|-----------|-------------|----------------|-----------|
| `half_precision` | `"false"` | Parsed as bool, then validated | main.py:308 |
| `test_time_augmentation` | `"false"` | Parsed as bool | main.py:309 |
| `class_agnostic_nms` | `"false"` | Parsed as bool | main.py:310 |
| `return_image` | `"false"` | Parsed as bool | main.py:312 |
| `show_labels` | `"true"` | Parsed as bool | main.py:313 |
| `show_confidence` | `"true"` | Parsed as bool | main.py:314 |

---

## Priority Order (Highest to Lowest)

1. **API Call Parameter** (if provided by user)
   - Example: `-F "confidence_threshold=0.25"`
   - Applied at: inference.py:357 (before the `or`)

2. **Environment Variable** (`.env` file)
   - Example: `SAI_CONFIDENCE_THRESHOLD=0.20` in `.env`
   - Loaded at: Startup by Pydantic BaseSettings
   - Applied at: inference.py:357 (after the `or`)

3. **Hardcoded Default** (in config.py)
   - Example: `default=0.13`
   - Loaded at: Startup if `.env` variable not found
   - Applied at: inference.py:357 (after the `or`)

---

## Code Validation

### Test 1: Confirm `or` operator behavior

**Code**: `inference.py:357`
```python
confidence = confidence_threshold or settings.confidence_threshold
```

**Python behavior**:
```python
# If confidence_threshold is None (falsy)
None or 0.13  # Returns 0.13

# If confidence_threshold is 0.25 (truthy)
0.25 or 0.13  # Returns 0.25

# Edge case: If user provides 0
0 or 0.13  # Returns 0.13 (bug - would ignore user's 0)
```

**Validation**: This is the actual code in production (line 357).

---

### Test 2: Confirm settings loading

**Code**: `config.py:1-15`
```python
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Force load .env file before settings initialization
load_dotenv(override=True)  # ← Line 13

class Settings(BaseSettings):
    confidence_threshold: float = Field(default=0.13, alias="SAI_CONFIDENCE_THRESHOLD")

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
```

**Validation**:
- `.env` is loaded explicitly at line 13
- Pydantic reads environment variables
- `alias="SAI_CONFIDENCE_THRESHOLD"` maps env var to field

---

### Test 3: Verify actual values in production

**Check current production defaults**:
```bash
# Production config
cat /etc/sai-inference/production.env | grep -E "SAI_CONFIDENCE|SAI_IOU"
```

**Result** (from production):
```bash
SAI_CONFIDENCE_THRESHOLD=0.13
SAI_IOU_THRESHOLD=0.4
```

These override the hardcoded defaults at startup.

---

## Summary

**Answer to "Where do defaults come from?"**

1. **NOT directly from `.env`** - there's a cascade
2. **Priority**: User API param > `.env` var > Hardcoded default
3. **Applied at**: `inference.py:356-360` using `or` operator
4. **Settings loaded**: Once at startup by Pydantic BaseSettings
5. **Runtime behavior**: Each API call checks parameter, falls back to `settings.*`

**Key Code Locations**:
- **API endpoint defaults**: `main.py:301-315` (mostly `None`)
- **Actual default application**: `inference.py:356-360` (the `or` operator)
- **Settings definition**: `config.py:25-42` (Pydantic + `.env`)
- **Settings loading**: `config.py:13` (`load_dotenv()`)

The `.env` file provides **tier 3 defaults** (startup configuration), but **tier 1** (API call) and **tier 2** (inference engine fallback) have precedence over it.
