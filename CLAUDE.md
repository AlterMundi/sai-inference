# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAI Inference Service is a high-performance FastAPI-based REST API for fire and smoke detection using YOLOv8 models. It's designed as a drop-in replacement for Ollama in n8n workflows, providing real-time inference with WebSocket support, and model hot-swapping.

## Key Architecture

### Core Components

1. **FastAPI Application** (`src/main.py`): Main REST API with endpoints for inference, model management, and n8n integration
2. **Inference Engine** (`src/inference.py`): YOLO model management and inference execution with caching
3. **Configuration** (`src/config.py`): Pydantic settings management with environment variable support
4. **Models** (`src/models.py`): Pydantic data models for API requests/responses

### SAINet2.1 Reference Parameters
The service is optimized for SAINet2.1 model with these critical parameters:
- **Input Resolution**: 1920px (reference: `imgsz=1920`)
- **Confidence Threshold**: 0.15 (reference: `conf=0.15`)
- **IOU Threshold**: 0.7 (reference: `iou=0.7`)
- **Model Path**: `models/last.pt`

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

# Format code
black src/

# Lint code
ruff src/
```

## API Endpoints

- **Health**: `GET /api/v1/health` - Service health and metrics
- **File Upload**: `POST /api/v1/infer` - Primary inference endpoint (multipart/form-data)
- **Base64 Inference**: `POST /api/v1/infer/base64` - JSON inference endpoint
- **Batch Processing**: `POST /api/v1/infer/batch` - Multiple images
- **Models**: `GET /api/v1/models`, `POST /api/v1/models/load`, `POST /api/v1/models/switch`

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

Key environment variables (`.env` file):
- `SAI_DEVICE`: cpu/cuda (GPU support)
- `SAI_MODEL_DIR`: Model directory path
- `SAI_DEFAULT_MODEL`: Default model filename
- `SAI_CONFIDENCE`: Detection confidence (0.15 for SAINet2.1)
- `SAI_INPUT_SIZE`: Input resolution (1920 for SAINet2.1)
- `SAI_API_KEY`: Optional API key for authentication

## Model Management

Models are stored in the `models/` directory. The service supports:
- Hot-swapping models without restart
- Multiple models loaded simultaneously
- Dynamic model switching via API
- Automatic device selection (CPU/GPU)

## Development Tips

1. **Model Location**: Place your SAI model as `models/last.pt`
2. **Performance**: Service handles ~50-100ms inference on CPU (fresh inference each time for critical accuracy)
3. **Batch Processing**: Supports up to 10 images in parallel
4. **Memory**: Requires ~2GB with model loaded
5. **Alert Levels**: Automatically determines severity based on detection counts and confidence

## Testing Workflow

```bash
# 1. Start service
python run.py

# 2. Run test suite
python tests/test_service.py

# 3. Check logs
tail -f logs/sai-inference.log

# 4. Monitor metrics
curl http://localhost:9090/metrics
```