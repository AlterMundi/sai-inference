# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAI Inference Service is a high-performance FastAPI-based REST API for fire and smoke detection using YOLOv8 models. It's designed as a drop-in replacement for Ollama in n8n workflows, providing real-time inference with WebSocket support, and model hot-swapping.

## Key Architecture

### Core Components

1. **FastAPI Application** (`src/main.py`): Main REST API with endpoints for inference, model management, and n8n integration
2. **Inference Engine** (`src/inference.py`): YOLO model management and inference execution
3. **Mosaic Inference** (`src/inference_mosaic.py`): Processes large images by splitting into overlapping 640x640 crops
4. **Configuration** (`src/config.py`): Pydantic settings management with environment variable support
5. **Models** (`src/models.py`): Pydantic data models for API requests/responses
6. **Daily Test Service** (`src/daily_test.py`): Automated testing system for end-to-end validation

### Model Specifications
- **Model Format**: YOLOv8s architecture (116MB `last.pt` file)
- **Detection Classes**: 2 classes - `0`: smoke, `1`: fire
- **Input Resolution**: 864px optimized (configurable via `SAI_INPUT_SIZE`)
- **Confidence Threshold**: 0.13 production default (was 0.15 in reference)
- **IOU Threshold**: 0.4 production default (was 0.7 in reference)

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

# Batch process images from directory
python scripts/process_images.py /path/to/images/

# Test n8n integration
./scripts/test_n8n_integration.sh

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

### System
- **Health Check**: `GET /api/v1/health` - Service health and metrics
- **Metrics**: `GET /metrics` - Prometheus metrics (port 9090)

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

## Environment Configuration

### Core Settings (`.env` file)
```bash
# Service Configuration
SAI_HOST=0.0.0.0           # Bind address
SAI_PORT=8888              # Service port
SAI_DEVICE=cpu             # cpu/cuda/cuda:0 for GPU
SAI_LOG_LEVEL=INFO         # DEBUG/INFO/WARNING/ERROR

# Model Configuration
SAI_MODEL_DIR=models       # Model directory path
SAI_DEFAULT_MODEL=last.pt  # Default model filename
SAI_CONFIDENCE=0.13        # Detection confidence threshold
SAI_IOU_THRESHOLD=0.4      # NMS IoU threshold
SAI_INPUT_SIZE=864         # Input resolution (int or "height,width")
SAI_MAX_DETECTIONS=100     # Maximum detections per image

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
tail -f logs/sai-inference.log

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

### Directory Structure
- **Production Install**: `/opt/sai-inference/`
- **Configuration**: `/etc/sai-inference/`
- **Logs**: `/var/log/sai-inference/`
- **Service User**: `service` (non-root)