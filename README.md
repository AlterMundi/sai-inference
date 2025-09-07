# SAI Inference Service

High-performance inference service for SAI fire/smoke detection, designed as a drop-in replacement for Ollama in n8n workflows.

## Features

- **FastAPI REST API** with automatic documentation
- **n8n Integration** via dedicated webhook endpoint
- **WebSocket support** for real-time inference
- **Model hot-swapping** without service restart
- **Response caching** for improved performance
- **Batch processing** for multiple images
- **Docker & SystemD** deployment options
- **Production-ready** with automated installation scripts

## Repository Structure

```
sai-inference/
├── src/                      # Core application code
├── config/                   # Configuration templates
├── deployment/               # Installation & deployment scripts
├── scripts/                  # Utility scripts & batch tools
├── tests/                    # Test suite
├── docs/                     # Documentation
├── examples/                 # Code examples & n8n workflows
└── models/                   # AI model storage
```

## Quick Start

### 1. Development Setup

```bash
# Quick setup script (recommended)
./deployment/setup.sh

# Or manual setup:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p models
# Place your model file as 'last.pt' in the models/ directory
```

### 2. Run Service

```bash
# Development
python run.py

# Production installation (automated)
sudo ./deployment/install.sh

# Test the service
python tests/test_service.py
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8888/api/v1/health
```

### Single Image Inference
```bash
curl -X POST http://localhost:8888/api/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_data"}'
```

### File Upload
```bash
curl -X POST http://localhost:8888/api/v1/infer/file \
  -F "file=@image.jpg"
```

### n8n Webhook
```bash
curl -X POST http://localhost:8888/webhook/sai \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_data", "callback_url": "https://your-n8n/webhook"}'
```

## n8n Integration

### HTTP Request Node Configuration

1. **Method**: POST
2. **URL**: `http://your-server:8888/api/v1/infer`
3. **Body Type**: JSON
4. **Body**:
```json
{
  "image": "{{$binary.data}}",
  "confidence_threshold": 0.45,
  "return_image": true
}
```

### Webhook Node Configuration

1. **Webhook URL**: `http://your-server:8888/webhook/sai`
2. **HTTP Method**: POST
3. **Response Mode**: "On Received"

### Example n8n Workflow

```json
{
  "nodes": [
    {
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "fire-detection",
        "responseMode": "onReceived",
        "options": {}
      }
    },
    {
      "name": "SAI Inference",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8888/api/v1/infer",
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "image",
              "value": "={{$json.image}}"
            }
          ]
        }
      }
    },
    {
      "name": "Alert on Fire",
      "type": "n8n-nodes-base.if",
      "parameters": {
        "conditions": {
          "boolean": [
            {
              "value1": "={{$json.has_fire}}",
              "value2": true
            }
          ]
        }
      }
    }
  ]
}
```

## Configuration

### Environment Variables

- `SAI_HOST`: Service host (default: 0.0.0.0)
- `SAI_PORT`: Service port (default: 8888)
- `SAI_DEVICE`: Compute device (cpu/cuda)
- `SAI_MODEL_DIR`: Model directory path
- `SAI_DEFAULT_MODEL`: Default model filename
- `SAI_CONFIDENCE`: Detection confidence threshold
- `SAI_API_KEY`: Optional API key for authentication
- `SAI_REDIS_URL`: Redis URL for distributed caching

### Model Management

```bash
# List models
curl http://localhost:8888/api/v1/models

# Load new model
curl -X POST http://localhost:8888/api/v1/models/load?model_name=new_model.pt

# Switch model
curl -X POST http://localhost:8888/api/v1/models/switch?model_name=sai_v2.1.pt
```

## Performance

- **Inference Speed**: ~50-100ms per image (CPU)
- **Batch Processing**: Up to 10 images in parallel
- **Cache Hit Rate**: ~30% for typical workflows
- **Memory Usage**: ~2GB with model loaded
- **Concurrent Requests**: Handles 100+ simultaneous connections

## API Documentation

Interactive API documentation available at:
- Swagger UI: http://localhost:8888/api/v1/docs
- ReDoc: http://localhost:8888/api/v1/redoc

## Alert Levels

The service automatically determines alert levels:
- **Critical**: Multiple fires or high-confidence fire detection
- **High**: Fire detected
- **Medium**: Multiple smoke detections
- **Low**: Smoke detected
- **None**: No detections

## Development

### Project Structure

- **`src/`** - Core application modules (main.py, inference.py, config.py, models.py)
- **`deployment/`** - Production deployment scripts (install.sh, setup.sh, uninstall.sh)
- **`scripts/`** - Utility scripts (process_images.py, test integrations)
- **`tests/`** - Test suite and integration tests  
- **`config/`** - Configuration templates for different environments
- **`docs/`** - API documentation and usage guides
- **`examples/`** - Code examples and n8n workflow templates

### Testing & Quality

```bash
# Run tests
python tests/test_service.py

# Test n8n integration
./scripts/test_n8n_integration.sh

# Format code
black src/

# Lint
ruff src/

# Process batch images
python scripts/process_images.py /path/to/images/
```

### Deployment

```bash
# Development setup
./deployment/setup.sh

# Production installation (requires sudo)
sudo ./deployment/install.sh

# Uninstall service
sudo ./deployment/uninstall.sh
```

## License

GNU v3