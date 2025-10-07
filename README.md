# SAI Inference Service

High-performance inference service for SAI fire/smoke detection, designed as a drop-in replacement for Ollama in n8n workflows.

## Features

- **FastAPI REST API** with automatic documentation
- **n8n Integration** via HTTP Request node
- **PostgreSQL Database Logging** - Comprehensive execution tracking
- **Temporal Alert System** - Dual-window escalation for wildfire detection
- **Camera Analytics API** - Historical detection statistics and trends
- **Model hot-swapping** without service restart
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

### File Upload (Primary Method)
```bash
# Basic inference
curl -X POST http://localhost:8888/api/v1/infer \
  -F "file=@image.jpg"

# With optimized parameters
curl -X POST http://localhost:8888/api/v1/infer \
  -F "file=@image.jpg" \
  -F "confidence_threshold=0.13" \
  -F "return_image=true"
```

### Base64 Inference (Alternative)
```bash
curl -X POST http://localhost:8888/api/v1/infer/base64 \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_data", "confidence_threshold": 0.13}'
```

### Camera Analytics & Monitoring
```bash
# List active cameras
curl http://localhost:8888/api/v1/cameras?hours=24

# Get camera statistics
curl http://localhost:8888/api/v1/cameras/cam-id/stats?hours=24

# View alert summary
curl http://localhost:8888/api/v1/alerts/summary?hours=24

# Escalation statistics
curl http://localhost:8888/api/v1/alerts/escalation-stats?hours=24

# Prometheus metrics
curl http://localhost:8888/metrics
```

### Advanced Parameters (New Features)
```bash
# Fire-only detection with GPU acceleration
curl -X POST http://localhost:8888/api/v1/infer \
  -F "file=@image.jpg" \
  -F "detection_classes=[1]" \
  -F "half_precision=true"

# Maximum accuracy mode for critical analysis
curl -X POST http://localhost:8888/api/v1/infer \
  -F "file=@image.jpg" \
  -F "test_time_augmentation=true" \
  -F "class_agnostic_nms=true"
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

1. **API URL**: `http://your-server:8888/api/v1/infer`
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

**Core Configuration**:
- `SAI_HOST`: Service host (default: 0.0.0.0)
- `SAI_PORT`: Service port (default: 8888)
- `SAI_DEVICE`: Compute device (cpu/cuda)
- `SAI_MODEL_DIR`: Model directory path
- `SAI_DEFAULT_MODEL`: Default model filename (default: last.pt)

**Optimized Detection Parameters** (Production Defaults):
- `SAI_CONFIDENCE_THRESHOLD`: Detection sensitivity (default: 0.39)
- `SAI_IOU_THRESHOLD`: Overlap handling (default: 0.1)
- `SAI_INPUT_SIZE`: Image processing resolution (default: 864)
- `SAI_DETECTION_CLASSES`: Filter classes (default: [0] for smoke-only)

**Database & Analytics**:
- `SAI_DATABASE_URL`: PostgreSQL connection string (required for temporal alerts)
- `SAI_WILDFIRE_HIGH_THRESHOLD`: High alert threshold (default: 0.7)
- `SAI_WILDFIRE_LOW_THRESHOLD`: Low alert threshold (default: 0.3)
- `SAI_ESCALATION_HOURS`: Critical escalation window (default: 3)
- `SAI_ESCALATION_MINUTES`: High escalation window (default: 30)
- `SAI_PERSISTENCE_COUNT`: Detections needed for escalation (default: 3)

**Optional**:
- `SAI_API_KEY`: API authentication key
- `SAI_LOG_LEVEL`: Logging verbosity (default: INFO)
- `SAI_ENABLE_METRICS`: Enable Prometheus metrics (default: true)

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
- **Memory Usage**: ~2GB with model loaded
- **Concurrent Requests**: Handles 100+ simultaneous connections

## Documentation

### API Documentation
All endpoints are documented in the codebase with comprehensive examples in the [docs/](docs/) directory:

- **[API_USAGE_GUIDE.md](docs/API_USAGE_GUIDE.md)** - Comprehensive API usage examples
- **[API_ARGUMENTS_REFERENCE.md](docs/API_ARGUMENTS_REFERENCE.md)** - Complete parameter reference
- **[DATABASE_VALIDATION.md](docs/DATABASE_VALIDATION.md)** - Database schema and validation procedures
- **[INFERENCE_PROCEDURE.md](docs/INFERENCE_PROCEDURE.md)** - Internal inference workflow
- **[DEFAULT_VALUES_FLOW.md](docs/DEFAULT_VALUES_FLOW.md)** - Parameter precedence and defaults
- **[EXAMPLE_OUTPUTS.md](docs/EXAMPLE_OUTPUTS.md)** - Sample API responses

For development guidance, see [CLAUDE.md](CLAUDE.md) which contains detailed architecture and operational procedures.

## Database & Logging System

### PostgreSQL Integration

The service logs ALL inference executions to PostgreSQL database `sai_dashboard` in table `camera_detections`:

**What's Logged**:
- Every API call (including zero-detection results)
- Full detection details (bounding boxes, confidence scores)
- Alert state transitions (base → final with escalation reasons)
- Performance metrics (processing time, model inference time)
- Model configuration (version, thresholds, classes)
- Request context (source, n8n workflow/execution IDs)

**Database Configuration**:
```bash
SAI_DATABASE_URL=postgresql://sai_user:password@localhost/sai_dashboard?sslmode=disable
```

**Verify Database**:
```bash
# Check connection and recent records
sudo -u postgres psql -d sai_dashboard -c "SELECT COUNT(*), MAX(created_at) FROM camera_detections;"

# View escalation events
sudo -u postgres psql -d sai_dashboard -c "SELECT * FROM camera_detections WHERE escalation_reason IS NOT NULL ORDER BY created_at DESC LIMIT 5;"
```

### Alert System Architecture

**Dual-Mode Operation**:
1. **Basic Mode** (no `camera_id` in request):
   - Single detection analysis only
   - No database logging
   - Returns: `"none"`, `"low"`, `"high"`

2. **Enhanced Mode** (with `camera_id` in request):
   - Temporal pattern analysis across time windows
   - Complete database logging
   - Returns: `"none"`, `"low"`, `"high"`, `"critical"`

**Alert Level Determination**:
- **None**: No detections
- **Low**: Detection confidence < 0.7
- **High**: Detection confidence ≥ 0.7 OR escalated from low (3+ in 30min)
- **Critical**: 3+ high-confidence detections in 3-hour window

**Temporal Escalation Windows**:
- **Low → High**: 30-minute window (3+ detections = immediate threat)
- **High → Critical**: 3-hour window (3+ high-confidence = sustained threat)

**Production Metrics** (typical):
- ~5,000 executions/day across all cameras
- Detection rate: 0.05-0.1% (high selectivity)
- Escalation rate: <0.1% (only persistent patterns)
- Average latency: 70-90ms per inference

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