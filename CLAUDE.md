# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAI Inference Service is a high-performance FastAPI-based REST API for fire and smoke detection using YOLOv8 models. It's designed as a drop-in replacement for Ollama in n8n workflows, providing real-time inference with WebSocket support, model hot-swapping, and response caching.

## Key Architecture

### Core Components

1. **FastAPI Application** (`src/main.py`): Main REST API with endpoints for inference, model management, and n8n integration
2. **Inference Engine** (`src/inference.py`): YOLO model management and inference execution with caching
3. **Configuration** (`src/config.py`): Pydantic settings management with environment variable support
4. **Models** (`src/models.py`): Pydantic data models for API requests/responses

### SAINet2.1 Reference Parameters
The service is optimized for SAINet2.1 model with these critical parameters from `/mnt/n8n-data/SAINet/SAINet2.1/inf_yolo11m_SAINet2.1.py`:
- **Input Resolution**: 1024px (reference: `imgsz=1024`)
- **Confidence Threshold**: 0.15 (reference: `conf=0.15`)
- **IOU Threshold**: 0.7 (reference: `iou=0.7`)
- **Model Path**: `/mnt/n8n-data/SAINet/SAINet2.1/SAINet2.1_130epochs/weights/last.pt`

## Essential Commands

### Development Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Quick setup (creates venv, installs deps, copies models)
./setup.sh

# Create models directory and copy SAI model
mkdir -p models
cp /mnt/n8n-data/SAINet_v1.0/datasets/D-Fire/SAINet2.1/best.pt models/sai_v2.1.pt
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
python test_service.py

# Test specific endpoints
curl http://localhost:8888/api/v1/health
curl -X POST http://localhost:8888/api/v1/infer/file -F "file=@image.jpg"

# Format code
black src/

# Lint code
ruff src/
```

## API Endpoints

- **Health**: `GET /api/v1/health` - Service health and metrics
- **Inference**: `POST /api/v1/infer` - Single image inference (base64)
- **File Upload**: `POST /api/v1/infer/file` - Direct file upload
- **Batch**: `POST /api/v1/infer/batch` - Multiple images
- **n8n Webhook**: `POST /webhook/sai` - n8n integration endpoint
- **WebSocket**: `WS /api/v1/ws` - Real-time inference
- **Models**: `GET /api/v1/models`, `POST /api/v1/models/load`
- **API Docs**: `http://localhost:8888/api/v1/docs` (Swagger UI)

## n8n Integration

The service acts as a drop-in replacement for Ollama in n8n workflows. **All integration happens through n8n's built-in HTTP Request node** - no custom n8n node development is required.

### Why HTTP Request Node (Not Custom Node):
- Creating a custom n8n node requires a separate development process (TypeScript, npm package, etc.)
- HTTP Request node is n8n's standard way to integrate with external REST APIs
- Avoids deployment complexity (custom nodes need installation in n8n environment)
- The service provides REST endpoints specifically designed for HTTP Request node usage

### Integration Endpoints:
1. **Standard API** (`/api/v1/infer`): Clean JSON endpoint for HTTP Request node
   - Expects: `{"image": "base64", "confidence_threshold": 0.45}`
   - Returns: Standard inference response with detections

2. **n8n Adapter** (`/webhook/sai`): Handles n8n's various data wrapper formats
   - Automatically unwraps n8n's binary/json structures
   - Extracts workflow metadata (workflow_id, execution_id)
   - Returns n8n-friendly response with success flags

### Typical n8n Workflow:
1. **Image Source** (Webhook, File, Camera, etc.)
2. **HTTP Request Node** → Calls SAI Inference Service
   - Method: POST
   - URL: `http://localhost:8888/api/v1/infer`
   - Body: Image data as base64
3. **IF Node** → Checks `has_fire` or `has_smoke` flags
4. **Alert/Action Nodes** → Based on detection results

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
- `SAI_REDIS_URL`: Redis URL for distributed caching

## Model Management

Models are stored in the `models/` directory. The service supports:
- Hot-swapping models without restart
- Multiple models loaded simultaneously
- Dynamic model switching via API
- Automatic device selection (CPU/GPU)

## Development Tips

1. **Model Location**: Primary SAI models are at `/mnt/n8n-data/SAINet_v1.0/`
2. **Performance**: Service handles ~50-100ms inference on CPU, with caching for repeated requests
3. **Batch Processing**: Supports up to 10 images in parallel
4. **Memory**: Requires ~2GB with model loaded
5. **Alert Levels**: Automatically determines severity based on detection counts and confidence

## Testing Workflow

```bash
# 1. Start service
python run.py

# 2. Run test suite
python test_service.py

# 3. Check logs
tail -f logs/sai-inference.log

# 4. Monitor metrics
curl http://localhost:9090/metrics
```