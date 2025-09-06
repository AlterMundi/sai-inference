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

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create models directory
mkdir -p models
```

### 2. Copy Model

```bash
# Copy the best SAINet2.1 model
cp /mnt/n8n-data/SAINet_v1.0/datasets/D-Fire/SAINet2.1/best.pt models/sai_v2.1.pt
```

### 3. Run Service

```bash
# Development
python run.py

# Production with SystemD
sudo cp sai-inference.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sai-inference
sudo systemctl start sai-inference

# Docker
docker-compose -f docker/docker-compose.yml up -d
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

```bash
# Run tests
pytest tests/

# Format code
black src/

# Lint
ruff src/
```

## License

GNU v3