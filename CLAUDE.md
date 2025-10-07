# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAI Inference Service is a high-performance FastAPI-based REST API for fire and smoke detection using YOLOv8 models. It's designed as a drop-in replacement for Ollama in n8n workflows, providing real-time inference and model hot-swapping.

## Key Architecture

### Core Components

1. **FastAPI Application** (`src/main.py`): Main REST API with endpoints for inference, model management, and n8n integration
2. **Inference Engine** (`src/inference.py`): YOLO model management and inference execution
3. **Mosaic Inference** (`src/inference_mosaic.py`): Processes large images by splitting into overlapping 640x640 crops
4. **Configuration** (`src/config.py`): Pydantic settings management with environment variable support
5. **Models** (`src/models.py`): Pydantic data models for API requests/responses
6. **Daily Test Service** (`src/daily_test.py`): Automated testing system for end-to-end validation
7. **CLI Tool** (`sai-inference.sh`): Command-line wrapper for quick image analysis via API

### Model Specifications
- **Model Format**: YOLOv8s architecture (39MB `last.pt` file)
- **Detection Classes**: 2 classes - `0`: smoke, `1`: fire
- **Default Configuration**: Smoke-only detection (`SAI_DETECTION_CLASSES=[0]`) for wildfire early warning
- **Input Resolution**: 864px optimized (configurable via `SAI_INPUT_SIZE`)
- **Production Thresholds**: confidence=0.39, iou=0.1 (optimized for wildfire detection)
- **Threshold Sources**: Environment defaults, overridable per API call

### SystemD Integration
The service includes proper systemd integration:
- **Watchdog Support**: Sends keepalive notifications every 30s (`WatchdogSec=60`)
- **Service Type**: `Type=notify` for proper startup/shutdown handling
- **Health Monitoring**: Automatic restart if watchdog timeout occurs
- **Graceful Shutdown**: Proper cleanup on service stop/restart

## Essential Commands

### Development Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Quick setup (creates venv, installs deps, copies models)
./deployment/setup.sh

# Download and setup model
mkdir -p models
curl -LO https://github.com/AlterMundi/sai-inference/releases/download/v0.1/last.pt
mv last.pt models/
```

### Running the Service
```bash
# Development mode
python run.py

# Production with uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8888

# Docker deployment
docker-compose -f docker/docker-compose.yml up -d

# SystemD service
sudo systemctl start sai-inference
sudo systemctl status sai-inference
sudo journalctl -u sai-inference -f
```

### Testing
```bash
# Run comprehensive test suite
python tests/test_service.py

# Test specific endpoints
curl http://localhost:8888/api/v1/health
curl -X POST http://localhost:8888/api/v1/infer -F "file=@image.jpg"

# CLI tool (simple inference)
sai-inference --health                    # Check API status
sai-inference image.jpg                   # Analyze single image
sai-inference /path/to/images/            # Batch process directory
sai-inference -c 0.5 -d 0 image.jpg       # Custom confidence, smoke-only

# Batch process images from directory (Python)
python scripts/process_images.py /path/to/images/

# Test n8n integration
./scripts/test_n8n_integration.sh

# Test new analytics endpoints
curl http://localhost:8888/api/v1/cameras                           # List active cameras
curl http://localhost:8888/api/v1/cameras/cam-id/stats              # Camera statistics
curl http://localhost:8888/api/v1/alerts/summary?hours=24           # Alert summary
curl http://localhost:8888/api/v1/alerts/escalation-stats           # Escalation analysis
curl http://localhost:8888/metrics                                   # Prometheus metrics

# Run daily test service
python src/daily_test.py --config config/daily-test.env

# Format code
black src/

# Lint code
ruff src/
```

## API Endpoints

### Core Inference
- **File Upload**: `POST /api/v1/infer` - Primary inference endpoint (multipart/form-data)
- **Base64 Inference**: `POST /api/v1/infer/base64` - JSON inference endpoint
- **Batch Processing**: `POST /api/v1/infer/batch` - Multiple images
- **Mosaic Inference**: `POST /api/v1/infer/mosaic` - Large image processing with 640x640 crops

### Model Management
- **List Models**: `GET /api/v1/models` - Get available and loaded models
- **Load Model**: `POST /api/v1/models/load` - Load a new model
- **Switch Model**: `POST /api/v1/models/switch` - Switch active model

### Camera Analytics API
- **List Cameras**: `GET /api/v1/cameras?hours=24` - List all cameras with recent activity
- **Camera Stats**: `GET /api/v1/cameras/{camera_id}/stats?hours=24` - Detection statistics for specific camera
- **Camera Detections**: `GET /api/v1/cameras/{camera_id}/detections?minutes=180&min_confidence=0.3` - Recent detections with filters
- **Camera Escalations**: `GET /api/v1/cameras/{camera_id}/escalations?hours=24` - Escalation events for camera

### Alert History API
- **Recent Alerts**: `GET /api/v1/alerts/recent?limit=100&camera_id=optional` - Recent alerts across cameras
- **Alert Summary**: `GET /api/v1/alerts/summary?hours=24&camera_id=optional` - Aggregated alert statistics
- **Escalation Stats**: `GET /api/v1/alerts/escalation-stats?hours=24&camera_id=optional` - Escalation analysis

### System & Monitoring
- **Health Check**: `GET /api/v1/health` - Service health and metrics
- **Prometheus Metrics**: `GET /metrics` - Prometheus-compatible metrics endpoint

## n8n Integration

The service acts as a drop-in replacement for Ollama in n8n workflows. **All integration happens through n8n's built-in HTTP Request node** - no custom n8n node development is required.

### Why HTTP Request Node (Not Custom Node):
- Creating a custom n8n node requires a separate development process (TypeScript, npm package, etc.)
- HTTP Request node is n8n's standard way to integrate with external REST APIs
- Avoids deployment complexity (custom nodes need installation in n8n environment)
- The service provides REST endpoints specifically designed for HTTP Request node usage

### Integration Endpoints:
1. **File Upload API** (`/api/v1/infer`): Primary endpoint for direct file uploads
   - Method: POST with multipart/form-data
   - File parameter: `file` (binary image data)
   - Optional parameters: `confidence_threshold`, `iou_threshold`, `return_image`
   - Returns: Standard inference response with detections

2. **Base64 API** (`/api/v1/infer/base64`): JSON endpoint for base64 image data
   - Expects: `{"image": "base64_encoded_image", "confidence_threshold": 0.15}`
   - Returns: Standard inference response with detections


### Typical n8n Workflows:

#### Option 1: File Upload (Recommended)
1. **Image Source** (Webhook, File, Camera, etc.)
2. **HTTP Request Node** → Calls SAI Inference Service
   - Method: POST
   - URL: `http://localhost:8888/api/v1/infer`
   - Body Type: Form-Data Binary
   - File parameter: `file` with binary image data
3. **IF Node** → Checks `has_fire` or `has_smoke` flags
4. **Alert/Action Nodes** → Based on detection results

#### Option 2: Base64 JSON (Alternative)
1. **Image Source** (Webhook, File, Camera, etc.)
2. **Convert to Base64** (if needed)
3. **HTTP Request Node** → Calls SAI Inference Service
   - Method: POST
   - URL: `http://localhost:8888/api/v1/infer/base64`
   - Body: JSON with `{"image": "base64_data", "confidence_threshold": 0.15}`
4. **IF Node** → Checks `has_fire` or `has_smoke` flags
5. **Alert/Action Nodes** → Based on detection results

### Response Format:
- Structured JSON with `has_fire`, `has_smoke` boolean flags
- Detection array with bounding boxes and confidence scores
- Alert levels for severity-based routing
- Compatible with n8n's expression syntax: `{{$json.has_fire}}`

## Database & Logging Architecture

### PostgreSQL Database Configuration

**Production Database**: `sai_dashboard` (shared with n8n services)
**Table**: `camera_detections` (owned by `sai_user`)
**Connection**: Async connection pool via asyncpg

```bash
# Database URL format
SAI_DATABASE_URL=postgresql://sai_user:password@localhost/sai_dashboard?sslmode=disable
```

### Database Schema Overview

The `camera_detections` table provides comprehensive logging of ALL inference executions:

**Key Features**:
- **Complete Execution Logging**: Every API call is recorded (including zero-detection results)
- **Temporal Alert Tracking**: Dual-window escalation system for wildfire detection
- **Performance Metrics**: Processing times, model inference metrics
- **Spatial Data**: JSONB storage of bounding boxes and detection details
- **Alert State Tracking**: Base alert → Final alert with escalation reasons

**Schema Structure** (25 columns):
```sql
-- Identification
id, camera_id, request_id, created_at

-- Detection Summary (fast aggregation)
detection_count, smoke_count, fire_count, max_confidence, avg_confidence

-- Alert System State (dual-mode tracking)
base_alert_level        -- Initial: none/low/high
final_alert_level       -- After temporal analysis: none/low/high/critical
escalation_reason       -- NULL | persistence_low | persistence_high | false_positive_pattern

-- Detection Details (JSONB for spatial analysis)
detections              -- Array of {class_id, class_name, confidence, bbox}

-- Performance Metrics
processing_time_ms, model_inference_time_ms, image_width, image_height

-- Model Configuration (reproducibility)
model_version, confidence_threshold, iou_threshold, detection_classes

-- Request Context (tracing)
source, n8n_workflow_id, n8n_execution_id, metadata
```

**Optimized Indexes** (11 indexes):
- B-tree: camera_id+created_at, alert levels, escalations
- GIN: JSONB detections for spatial queries
- Partial: High-confidence detections, cameras with detections

### Alert System Architecture

**Dual-Mode Operation**:
1. **Basic Mode** (no camera_id): Single detection analysis → "none"/"low"/"high"
2. **Enhanced Mode** (with camera_id): Temporal tracking + DB logging → "none"/"low"/"high"/"critical"

**Temporal Escalation Windows**:
- **Low → High**: 30-minute window (3+ detections → immediate threat)
- **High → Critical**: 3-hour window (3+ high-confidence → sustained threat)

**Escalation Tracking**:
- All escalations stored with `escalation_reason` field
- Integrity constraint: `base_alert_level != final_alert_level` requires reason
- Query-time analytics via Analytics API endpoints

### Production Database Verification

```bash
# Check database connection
sudo -u postgres psql -d sai_dashboard -c "SELECT application_name, state FROM pg_stat_activity WHERE application_name LIKE '%sai%';"

# Check recent records
sudo -u postgres psql -d sai_dashboard -c "SELECT COUNT(*), MAX(created_at) FROM camera_detections;"

# View escalation statistics
curl http://localhost:8888/api/v1/alerts/escalation-stats?hours=24
```

**Expected Metrics** (production baseline):
- ~5,000 executions/day across all cameras
- Detection rate: 0.05-0.1% (high selectivity, low false positives)
- Escalation rate: <0.1% (only persistent patterns)
- Average latency: 70-90ms per inference

## Environment Configuration

### Core Settings (`.env` file)
```bash
# Service Configuration
SAI_HOST=0.0.0.0           # Bind address
SAI_PORT=8888              # Service port
SAI_DEVICE=cpu             # cpu/cuda/cuda:0 for GPU
SAI_LOG_LEVEL=INFO         # DEBUG/INFO/WARNING/ERROR

# Database Configuration
SAI_DATABASE_URL=postgresql://sai_user:password@localhost/sai_dashboard?sslmode=disable

# Model Configuration
SAI_MODEL_DIR=models       # Model directory path
SAI_DEFAULT_MODEL=last.pt  # Default model filename
SAI_CONFIDENCE_THRESHOLD=0.39  # Detection confidence threshold (production optimized)
SAI_IOU_THRESHOLD=0.1      # NMS IoU threshold (lower = more overlapping boxes allowed)
SAI_INPUT_SIZE=864         # Input resolution (int or "height,width")
SAI_MAX_DETECTIONS=100     # Maximum detections per image

# Wildfire Detection - Smoke-only for early warning
SAI_DETECTION_CLASSES=[0]  # [0]=smoke-only, [1]=fire-only, [0,1]=both

# Optional Features
SAI_API_KEY=               # API authentication key (optional)
SAI_BATCH_SIZE=1           # Batch processing size
SAI_MAX_UPLOAD=52428800    # Max upload size (50MB)
```

### Daily Test Configuration (`config/daily-test.env`)
```bash
N8N_WEBHOOK_URL=           # n8n webhook endpoint
N8N_API_KEY=               # n8n authentication
IMAGE_DIR=/path/to/test    # Test images directory
ENABLED_TESTS=both,fire,smoke  # Test categories
```

## Model Management

Models are stored in the `models/` directory. The service supports:
- Hot-swapping models without restart
- Multiple models loaded simultaneously
- Dynamic model switching via API
- Automatic device selection (CPU/GPU)

## Development Tips

### Performance Characteristics
- **Inference Speed**: ~50-100ms per image on CPU
- **Batch Processing**: Up to 10 images in parallel
- **Memory Usage**: ~2GB with model loaded
- **Mosaic Processing**: 640x640 crops with 64px overlap for large images
- **YOLO Preprocessing**: Automatic letterboxing, normalization, tensor conversion

### Alert Level Logic
The service automatically determines severity:
- **Critical**: Multiple fires or high-confidence fire (>0.7)
- **High**: Fire detected
- **Medium**: Multiple smoke detections
- **Low**: Smoke detected
- **None**: No detections

### Common Issues & Solutions

1. **Model not loading**: Ensure `models/last.pt` exists and is a valid YOLOv8 model
2. **High memory usage**: Reduce `SAI_BATCH_SIZE` or use smaller input resolution
3. **Slow inference**: Enable GPU with `SAI_DEVICE=cuda` if available
4. **SystemD watchdog timeout**: Increase `WatchdogSec` in service file or disable watchdog

## Testing Workflow

```bash
# 1. Start service
python run.py

# 2. Run test suite
python tests/test_service.py

# 3. Check logs
tail -f /var/log/sai-inference/service.log

# 4. Monitor systemd service
sudo journalctl -u sai-inference -f

# 5. Test with sample image
curl -X POST http://localhost:8888/api/v1/infer \
  -F "file=@tests/images/fire/sample.jpg" \
  -F "confidence_threshold=0.13"
```

## Deployment Notes

### Production Installation
```bash
# Automated installation (creates service user, systemd service, etc.)
sudo ./deployment/install.sh

# Daily test service installation
sudo ./deployment/install-daily-test.sh

# Uninstall everything
sudo ./deployment/uninstall.sh
```

### Production Directory Structure
- **Install**: `/opt/sai-inference/` (service files, venv)
- **Config**: `/etc/sai-inference/production.env` (environment)
- **Logs**: `/var/log/sai-inference/service.log` (systemd output)
- **User**: `service` (non-root execution)