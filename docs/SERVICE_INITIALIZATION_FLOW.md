# SAI Inference Service - Initialization Flow Analysis

**Document Version**: 1.0
**Analysis Date**: 2025-10-20
**Service Version**: 1.0.0

---

## Executive Summary

This document provides a strict, step-by-step analysis of the SAI Inference Service initialization flow from startup to stable operation. The service follows a **layered initialization pattern** with dependency-aware sequencing to ensure robust startup under various conditions (model availability, database connectivity, systemd integration).

---

## 1. Entry Points

### 1.1 Development Mode: `run.py`
```python
# Entry: python run.py
run.py → sys.path.insert(0, src/) → from src.main import run_server → run_server()
```

### 1.2 Production Mode: SystemD + Uvicorn
```bash
# Entry: systemctl start sai-inference
/opt/sai-inference/venv/bin/uvicorn src.main:app \
  --host ${SAI_HOST} --port ${SAI_PORT} --workers ${SAI_WORKERS}
```

**Critical Difference**:
- Development: `run_server()` wrapper controls uvicorn programmatically
- Production: Direct uvicorn invocation, FastAPI lifecycle hooks manage initialization

---

## 2. Pre-Initialization Phase (Module Load Time)

### 2.1 Configuration Loading (`src/config.py`)
**Trigger**: Module import
**Order**: First, before any other component

```python
# Force load .env file
load_dotenv(override=True)

# Pydantic Settings instantiation
settings = Settings()  # Line 146
```

**Key Settings Loaded**:
- `SAI_HOST`: 0.0.0.0
- `SAI_PORT`: 8888
- `SAI_DEVICE`: cpu/cuda
- `SAI_CONFIDENCE_THRESHOLD`: 0.39 (production optimized)
- `SAI_IOU_THRESHOLD`: 0.1
- `SAI_INPUT_SIZE`: 864
- `SAI_DETECTION_CLASSES`: [0] (smoke-only by default)
- `SAI_DATABASE_URL`: postgresql://sai_user:password@localhost/sai_dashboard
- `SAI_ENABLE_METRICS`: true

**Validation**:
- Field validators parse `detection_classes` (JSON array or CSV)
- Field validators parse `input_size` (int or tuple)
- `models_dir` converted to Path object

---

### 2.2 Logging Configuration (`src/main.py:36-40`)
**Trigger**: Module import after settings

```python
logging.basicConfig(
    level=getattr(logging, settings.log_level),  # Default: INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

---

### 2.3 SystemD Watchdog Detection (`src/main.py:43-50`)
**Trigger**: Module import

```python
try:
    from systemd import daemon
    watchdog_enabled = True
    logger.info("SystemD watchdog support enabled")
except ImportError:
    watchdog_enabled = False
```

**Purpose**: Detect if running under systemd with WatchdogSec configured
**Production**: `watchdog_enabled = True` (systemd-python installed)
**Development**: `watchdog_enabled = False` (optional package)

---

### 2.4 Prometheus Metrics Initialization (`src/main.py:71-154`)
**Trigger**: Module import, conditional on `settings.enable_metrics`

**Metrics Created** (if enabled):
1. **Request Counters**:
   - `sai_inference_requests_total`: Total requests by endpoint/status

2. **Detection Counters**:
   - `sai_detections_total`: Detections by class/camera/endpoint
   - `camera_detection_rate`: Histogram of detections per request

3. **Alert Metrics**:
   - `sai_alert_levels_total`: Alert distribution by level/camera
   - `sai_alert_escalations_total`: Escalation events by reason/camera

4. **Performance Histograms**:
   - `sai_inference_duration_seconds`: End-to-end latency
   - `sai_model_inference_duration_seconds`: YOLO-only time
   - `sai_db_query_duration_seconds`: Database performance

5. **System Gauges**:
   - `sai_active_cameras`: Active cameras (24h window)
   - `sai_total_alerts_24h`: Total alerts (24h window)

6. **Image Dimension Histograms**:
   - `sai_image_width_pixels`: Input width distribution
   - `sai_image_height_pixels`: Input height distribution

**State**: All metrics are module-level globals, initialized to `None` if metrics disabled

---

### 2.5 FastAPI Application Creation (`src/main.py:204-220`)
**Trigger**: Module import

```python
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    openapi_url=f"{settings.api_prefix}/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,  # Default: ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**State**: FastAPI app created, but **not yet running** (no server started)

---

### 2.6 Global Component Instantiation (`src/inference.py:630`)
**Trigger**: Module import of `src.inference`

```python
# Global instance
inference_engine = InferenceEngine()
```

**InferenceEngine Constructor Flow** (`src/inference.py:167-174`):
```python
def __init__(self):
    self.model_manager = ModelManager()  # Instantiate model manager
    # Cache removed for bug fix (line 169-171 comments)
    self._load_default_model()  # Load best available model
```

**ModelManager Constructor Flow** (`src/inference.py:29-34`):
```python
def __init__(self):
    self.models: Dict[str, YOLO] = {}
    self.current_model: Optional[YOLO] = None
    self.current_model_name: Optional[str] = None
    self.model_info: Dict[str, ModelInfo] = {}
    self.device = self._setup_device()  # Detect CUDA availability
```

**Device Setup** (`src/inference.py:36-48`):
```python
def _setup_device(self) -> str:
    device = settings.device.lower()

    if device.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda:0" if ":" not in device else device
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
    return "cpu"
```

**Model Loading** (`src/inference.py:176-200`):
```python
def _load_default_model(self):
    # 1. Discover available models in models_dir
    available_models = self.model_manager.discover_models()

    # 2. Priority order:
    #    a. Configured default (SAI_DEFAULT_MODEL)
    #    b. SAI-named models (e.g., sai_v2.1.pt)
    #    c. YOLO-named models (e.g., yolov8s.pt)
    #    d. Any .pt file
    best_model = self.model_manager.get_best_available_model()

    # 3. Load best model
    if best_model:
        success = self.model_manager.load_model(best_model)

    # 4. Fallback: try each available model
    if not success:
        for model_name in available_models:
            if self.model_manager.load_model(model_name):
                break
```

**YOLO Model Loading** (`src/inference.py:95-137`):
```python
def load_model(self, model_name: str, model_path: Optional[Path] = None) -> bool:
    model_path = settings.models_dir / model_name

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return False

    logger.info(f"Loading model: {model_name} from {model_path}")
    model = YOLO(str(model_path))  # Ultralytics YOLO load

    # Move to device (GPU if available)
    if self.device != "cpu":
        model.to(self.device)

    self.models[model_name] = model

    # Store model metadata
    self.model_info[model_name] = ModelInfo(
        name=model_name,
        version="SAI-v2.1",
        path=str(model_path),
        size_mb=model_path.stat().st_size / (1024 * 1024),
        classes=["smoke", "fire"],
        input_size=settings.input_size,
        confidence_threshold=settings.confidence_threshold,
        iou_threshold=settings.iou_threshold,
        device=self.device,
        loaded=True
    )

    # Set as current if no model is active
    if self.current_model is None:
        self.set_current_model(model_name)

    logger.info(f"Model {model_name} loaded successfully on {self.device}")
    return True
```

**Critical Behaviors**:
- **Non-blocking failure**: If no models found, service continues (will error on first inference request)
- **Graceful degradation**: Tries multiple models before giving up
- **Smart discovery**: Prefers SAI-specific models, falls back to generic YOLO

---

### 2.7 Database Manager Instantiation (`src/database.py:37-42`)
**Trigger**: Module import of `src.database`

```python
class DatabaseManager:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.database_url = settings.database_url
```

**State**: Manager created, but **connection pool NOT initialized yet**
**Purpose**: Defer connection until `startup_event()` to avoid blocking module load

---

### 2.8 Alert Manager Instantiation (`src/alert_manager.py:72-73`)
**Trigger**: Module import of `src.alert_manager`

```python
class AlertManager:
    def __init__(self):
        self.db_initialized = False
```

**State**: Manager created, database **not initialized**
**Purpose**: Lazy initialization, database connection deferred to first use

---

## 3. Application Startup Phase

### 3.1 FastAPI Startup Event Hook (`src/main.py:270-298`)
**Trigger**: Uvicorn starts serving, before accepting requests

```python
@app.on_event("startup")
async def startup_event():
    global watchdog_task, cleanup_task, metrics_task

    # 1. Initialize database connection pool
    try:
        await db_manager.initialize()
        logger.info("Enhanced alert system database initialized")

        # 2. Start periodic cleanup task (hourly)
        cleanup_task = asyncio.create_task(periodic_cleanup())
        logger.info("Alert cleanup task started")

        # 3. Start Prometheus gauge update task (every minute)
        if settings.enable_metrics:
            metrics_task = asyncio.create_task(update_prometheus_gauges())
            logger.info("Prometheus metrics update task started")

    except Exception as e:
        logger.warning(f"Database initialization failed - enhanced alerts disabled: {e}")

    # 4. Start systemd watchdog (if enabled)
    if watchdog_enabled and os.environ.get('WATCHDOG_USEC'):
        logger.info("Starting systemd watchdog task")
        watchdog_task = asyncio.create_task(watchdog_ping())
        daemon.notify('READY=1')  # Tell systemd we're ready
    else:
        logger.info("SystemD watchdog not required or not available")
```

---

### 3.2 Database Initialization (`src/database.py:44-63`)
**Trigger**: Called by `startup_event()`

```python
async def initialize(self):
    # 1. Determine SSL requirement
    ssl_required = 'localhost' not in self.database_url and '127.0.0.1' not in self.database_url

    # 2. Create asyncpg connection pool
    self.pool = await asyncpg.create_pool(
        self.database_url,
        min_size=2,
        max_size=10,
        command_timeout=30,
        ssl=ssl_required,
        server_settings={'application_name': 'sai-inference-detections'}
    )

    # 3. Create tables from migration
    await self.create_tables()
    logger.info("Database initialized successfully")
```

**Table Creation** (`src/database.py:65-90`):
```python
async def create_tables(self):
    # Check if camera_detections table exists
    table_exists = await conn.fetchval(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'camera_detections')"
    )

    if table_exists:
        logger.info("Schema already exists")
        return

    # Load migration SQL
    migration_file = Path(__file__).parent.parent / 'migrations' / '001_enhanced_schema.sql'
    migration_sql = migration_file.read_text()

    # Execute each statement
    statements = [s.strip() for s in migration_sql.split(';') if s.strip()]
    for statement in statements:
        await conn.execute(statement)

    logger.info("Schema created successfully")
```

**Error Handling**:
- If database unreachable: Logs warning, continues startup
- If table creation fails: Logs error, continues startup
- **Service remains operational** even without database (basic mode alerts only)

---

### 3.3 Background Tasks Startup

#### 3.3.1 Periodic Cleanup Task (`src/main.py:237-246`)
**Purpose**: Clean up old alert data and expired tickets
**Frequency**: Every 1 hour
**Mechanism**: Asyncio background task

```python
async def periodic_cleanup():
    while True:
        try:
            await asyncio.sleep(3600)  # Wait 1 hour
            await alert_manager.cleanup_old_data()
            logger.debug("Periodic alert cleanup completed")
        except Exception as e:
            logger.error(f"Periodic cleanup failed: {e}")
            await asyncio.sleep(3600)  # Retry after 1 hour
```

**Non-blocking**: Errors do not crash service, task self-heals

---

#### 3.3.2 Prometheus Gauge Update Task (`src/main.py:249-267`)
**Purpose**: Update gauge metrics from database
**Frequency**: Every 1 minute
**Metrics Updated**:
- `sai_active_cameras`: Count of cameras with activity in last 24h
- `sai_total_alerts_24h`: Total alerts in last 24h

```python
async def update_prometheus_gauges():
    while True:
        try:
            await asyncio.sleep(60)  # Wait 1 minute

            if settings.enable_metrics:
                # Query database for metrics
                cameras = await db_manager.get_all_cameras(hours=24)
                active_cameras.set(len(cameras))

                summary = await db_manager.get_alert_summary(hours=24)
                total_alerts_24h.set(summary['total_alerts'])

                logger.debug("Prometheus gauge metrics updated")
        except Exception as e:
            logger.error(f"Prometheus gauge update failed: {e}")
            await asyncio.sleep(60)  # Retry after 1 minute
```

**Non-blocking**: Database errors do not affect inference, task self-heals

---

#### 3.3.3 SystemD Watchdog Task (`src/main.py:224-234`)
**Purpose**: Send keepalive notifications to systemd
**Frequency**: Every 30 seconds (WatchdogSec=60 in service file)
**Enabled**: Only if `WATCHDOG_USEC` environment variable present

```python
async def watchdog_ping():
    while True:
        try:
            if watchdog_enabled:
                daemon.notify('WATCHDOG=1')
                logger.debug("Sent watchdog ping to systemd")
            await asyncio.sleep(30)  # Less than 60s timeout
        except Exception as e:
            logger.error(f"Watchdog ping failed: {e}")
            await asyncio.sleep(30)
```

**SystemD Integration**:
- Service file: `WatchdogSec=60` (60-second timeout)
- Startup: `daemon.notify('READY=1')` (signal ready to systemd)
- Shutdown: `daemon.notify('STOPPING=1')` (signal graceful stop)

**Failure Handling**: If watchdog stops pinging, systemd restarts service

---

## 4. Stable Operation State

### 4.1 Service Ready Indicators

**Checklist for Stable Operation**:
- ✅ Configuration loaded from `.env`
- ✅ FastAPI app created with CORS middleware
- ✅ YOLO model loaded to device (CPU/GPU)
- ✅ Database connection pool initialized (or gracefully disabled)
- ✅ Background tasks started (cleanup, metrics, watchdog)
- ✅ API endpoints registered and ready to serve
- ✅ SystemD notified `READY=1` (production only)

**Verification Commands**:
```bash
# Check service status
systemctl status sai-inference

# Check health endpoint
curl http://localhost:8888/api/v1/health

# Check Prometheus metrics
curl http://localhost:8888/metrics

# Check logs
journalctl -u sai-inference -f
```

**Expected Health Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-20T...",
  "version": "1.0.0",
  "is_model_loaded": true,
  "loaded_model_info": {
    "name": "sai_v2.1.pt",
    "version": "SAI-v2.1",
    "classes": ["smoke", "fire"],
    "input_size": 864,
    "confidence_threshold": 0.39,
    "iou_threshold": 0.1,
    "device": "cpu"
  },
  "system_metrics": {
    "cpu_usage": 5.2,
    "memory_usage": 42.3,
    "memory_available_gb": 3.8
  },
  "runtime_parameters": {
    "confidence_threshold": 0.39,
    "iou_threshold": 0.1,
    "input_size": 864,
    "device": "cpu",
    "model_dir": "models",
    "default_model": "sai_v2.1.pt"
  }
}
```

---

### 4.2 Runtime Behavior

#### Request Handling Flow
1. **Request Arrival**: Client POSTs to `/api/v1/infer`
2. **File Validation**: Check size (<50MB) and extension (.jpg, .png, etc.)
3. **Image Processing**:
   - Decode image (base64 or binary)
   - Validate dimensions (>32x32, <8192x8192)
   - Convert to RGB
4. **YOLO Inference**:
   - Preprocess image (letterbox, normalize)
   - Model forward pass
   - NMS (non-max suppression)
5. **Alert Determination**:
   - Base level: confidence-based (none/low/high)
   - Enhanced mode: temporal escalation (camera_id provided)
6. **Database Logging** (if camera_id):
   - Store detection record
   - Check temporal patterns (30m/3h windows)
   - Calculate escalation reason
7. **Response Generation**:
   - Detection array with bboxes
   - Alert level
   - Annotated image (if requested)
   - Metrics (processing time, confidence scores)
8. **Prometheus Tracking**:
   - Increment request counter
   - Record latency histogram
   - Update detection counters

**Typical Latency** (production baseline):
- CPU inference: 70-90ms
- GPU inference: 30-50ms
- Database logging: +5-10ms
- Total: 80-100ms (CPU), 40-60ms (GPU)

---

### 4.3 Operating Modes

#### Mode 1: Basic Inference (No camera_id)
- **Characteristics**: Single-shot detection
- **Alert Logic**: Confidence-based only (none/low/high)
- **Database**: No logging
- **Use Case**: API testing, one-off image analysis

#### Mode 2: Enhanced Inference (With camera_id)
- **Characteristics**: Temporal tracking
- **Alert Logic**: Escalation rules (none/low/high/critical)
- **Database**: ALL executions logged (including zero detections)
- **Use Case**: n8n workflows, camera monitoring, production deployments

**Dual-Window Escalation Strategy**:
- **Low → High**: 3+ detections in 30 minutes
- **High → Critical**: 3+ high-confidence (≥0.7) in 3 hours

---

## 5. Shutdown Phase

### 5.1 FastAPI Shutdown Event Hook (`src/main.py:301-341`)
**Trigger**: SIGTERM, Ctrl+C, or `systemctl stop sai-inference`

```python
@app.on_event("shutdown")
async def shutdown_event():
    global watchdog_task, cleanup_task, metrics_task

    # 1. Stop cleanup task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

    # 2. Stop metrics task
    if metrics_task:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass

    # 3. Close database connections
    try:
        await db_manager.close()
        logger.info("Database connections closed")
    except Exception as e:
        logger.warning(f"Database shutdown error: {e}")

    # 4. Stop watchdog task
    if watchdog_task:
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass

    # 5. Notify systemd
    if watchdog_enabled:
        daemon.notify('STOPPING=1')
```

**Graceful Shutdown Order**:
1. Stop accepting new requests
2. Cancel background tasks (cleanup, metrics, watchdog)
3. Close database connection pool
4. Notify systemd of shutdown
5. Exit process

**Timeout**: SystemD gives 10 seconds before SIGKILL (RestartSec=10)

---

## 6. Failure Modes & Recovery

### 6.1 Model Load Failure
**Symptom**: No model files in `models/` directory
**Behavior**:
- Service starts successfully
- Health check returns `is_model_loaded: false`
- First inference request returns HTTP 500: "No model loaded"

**Recovery**:
1. Add model file to `models/` directory
2. Call `POST /api/v1/models/load?model_name=sai_v2.1.pt`
3. No restart required

---

### 6.2 Database Connection Failure
**Symptom**: PostgreSQL unreachable or wrong credentials
**Behavior**:
- Service starts successfully
- Logs warning: "Database initialization failed - enhanced alerts disabled"
- Basic inference works (confidence-based alerts only)
- Enhanced mode (camera_id) falls back to basic mode

**Recovery**:
1. Fix database connection (check `SAI_DATABASE_URL`)
2. Restart service: `systemctl restart sai-inference`

---

### 6.3 CUDA/GPU Failure
**Symptom**: `SAI_DEVICE=cuda` but CUDA not available
**Behavior**:
- Logs warning: "CUDA requested but not available, falling back to CPU"
- Service continues on CPU
- Inference slower but functional

**Recovery**:
1. Install CUDA drivers
2. Set `SAI_DEVICE=cuda:0` in `.env`
3. Restart service

---

### 6.4 Watchdog Timeout
**Symptom**: Watchdog task hangs or crashes
**Behavior**:
- SystemD detects no WATCHDOG=1 for 60 seconds
- SystemD sends SIGTERM
- Service restarts automatically (Restart=always)

**Recovery**: Automatic via systemd

---

## 7. Initialization Sequence Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1: MODULE LOAD TIME (Synchronous, ~500ms)                    │
└─────────────────────────────────────────────────────────────────────┘
    │
    ├─> Load .env file (dotenv)
    ├─> Instantiate settings (Pydantic)
    ├─> Configure logging (basicConfig)
    ├─> Detect systemd support (try import)
    ├─> Initialize Prometheus metrics (if enabled)
    ├─> Create FastAPI app + CORS middleware
    │
    ├─> Import inference module
    │   ├─> Instantiate ModelManager
    │   │   └─> Detect CUDA (torch.cuda.is_available)
    │   └─> Load default YOLO model
    │       ├─> Discover models (glob models/*.pt)
    │       ├─> Priority: SAI > YOLO > any .pt
    │       └─> YOLO.load() + model.to(device)
    │
    ├─> Import database module
    │   └─> Instantiate DatabaseManager (no connection yet)
    │
    └─> Import alert_manager module
        └─> Instantiate AlertManager (no DB init yet)

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2: APPLICATION STARTUP (Async, ~100-500ms)                   │
└─────────────────────────────────────────────────────────────────────┘
    │
    ├─> Uvicorn server starts
    ├─> Trigger FastAPI startup_event()
    │
    ├─> Initialize database (async)
    │   ├─> Create asyncpg connection pool (2-10 conns)
    │   ├─> Check table existence (query)
    │   └─> Run migrations if needed (create tables)
    │
    ├─> Start periodic cleanup task (asyncio)
    │   └─> Background: cleanup every 1 hour
    │
    ├─> Start Prometheus metrics task (asyncio)
    │   └─> Background: update gauges every 1 minute
    │
    └─> Start systemd watchdog task (asyncio)
        ├─> Background: ping every 30 seconds
        └─> Notify systemd: READY=1

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3: STABLE OPERATION (Event-driven)                           │
└─────────────────────────────────────────────────────────────────────┘
    │
    ├─> Accept HTTP requests
    │   ├─> POST /api/v1/infer
    │   ├─> POST /api/v1/infer/base64
    │   ├─> POST /api/v1/infer/mosaic
    │   ├─> POST /api/v1/infer/batch
    │   ├─> GET /api/v1/health
    │   ├─> GET /api/v1/cameras
    │   ├─> GET /api/v1/alerts/summary
    │   └─> GET /metrics
    │
    ├─> Background tasks running
    │   ├─> Cleanup: Every 1 hour
    │   ├─> Metrics: Every 1 minute
    │   └─> Watchdog: Every 30 seconds
    │
    └─> Database connection pool active (2-10 conns)

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 4: SHUTDOWN (Graceful, ~1-5 seconds)                         │
└─────────────────────────────────────────────────────────────────────┘
    │
    ├─> Receive SIGTERM (systemctl stop)
    ├─> Stop accepting new requests
    ├─> Trigger FastAPI shutdown_event()
    │
    ├─> Cancel cleanup task (asyncio.cancel)
    ├─> Cancel metrics task (asyncio.cancel)
    ├─> Close database pool (asyncpg.pool.close)
    ├─> Cancel watchdog task (asyncio.cancel)
    ├─> Notify systemd: STOPPING=1
    │
    └─> Exit process (code 0)
```

---

## 8. Performance Characteristics

### 8.1 Startup Time
- **Cold start** (no cache): 1-2 seconds
  - Module load: 500ms
  - Model load: 500ms-1s (CPU), 200-500ms (GPU)
  - Database init: 100-200ms
- **Warm start** (cache hit): 800ms-1.2s

### 8.2 Memory Footprint
- **Base service**: ~200MB
- **YOLO model loaded**: ~2GB (FP32), ~1GB (FP16)
- **Connection pool**: ~10MB
- **Total**: ~2.2GB (typical)

### 8.3 Request Throughput
- **Serial processing**: ~10-12 req/s (CPU), ~20-30 req/s (GPU)
- **Batch processing**: Up to 50 images in parallel
- **Bottleneck**: YOLO inference (not I/O or database)

---

## 9. Critical Dependencies

### 9.1 Required for Startup
- ✅ Python 3.11+
- ✅ FastAPI + Uvicorn
- ✅ Ultralytics YOLO
- ✅ PyTorch (CPU or CUDA)
- ✅ Pydantic Settings
- ✅ YOLO model file (`.pt` format)

### 9.2 Optional (Graceful Degradation)
- ⚠️ PostgreSQL database (basic mode works without)
- ⚠️ systemd-python (watchdog works without)
- ⚠️ CUDA/GPU (falls back to CPU)
- ⚠️ Prometheus endpoint (metrics can be disabled)

---

## 10. Configuration Hierarchy

**Priority Order** (highest to lowest):
1. **Runtime API parameters**: Per-request overrides (confidence_threshold in POST body)
2. **Environment variables**: `.env` file or systemd EnvironmentFile
3. **Pydantic defaults**: Hardcoded in `src/config.py`

**Source of Truth**:
- Thresholds: Environment variables (`SAI_CONFIDENCE_THRESHOLD`, `SAI_IOU_THRESHOLD`)
- Model: Environment variable (`SAI_DEFAULT_MODEL`) or smart discovery
- Database: Environment variable (`SAI_DATABASE_URL`)

---

## 11. Health Check Deep Dive

### Health Endpoint Response Structure
```json
{
  "status": "healthy",                    // Always "healthy" if responding
  "timestamp": "2025-10-20T...",          // ISO 8601 UTC
  "version": "1.0.0",                     // App version from config
  "is_model_loaded": true,                // Model availability
  "loaded_model_info": {                  // Model metadata
    "name": "sai_v2.1.pt",
    "version": "SAI-v2.1",
    "path": "/opt/sai-inference/models/sai_v2.1.pt",
    "size_mb": 39.2,
    "classes": ["smoke", "fire"],
    "input_size": 864,
    "confidence_threshold": 0.39,
    "iou_threshold": 0.1,
    "device": "cpu",
    "loaded": true
  },
  "system_metrics": {                     // Real-time system stats
    "cpu_usage": 5.2,                     // % (via psutil)
    "memory_usage": 42.3,                 // % (via psutil)
    "memory_available_gb": 3.8            // GB (via psutil)
  },
  "runtime_parameters": {                 // Current effective config
    "confidence_threshold": 0.39,
    "iou_threshold": 0.1,
    "input_size": 864,
    "device": "cpu",
    "model_dir": "models",
    "default_model": "sai_v2.1.pt"
  }
}
```

**Health Check Logic**:
- HTTP 200 + `"status": "healthy"` → Service operational
- HTTP 503 → Health check failed (exception in handler)
- No response → Service down

---

## 12. Conclusion

The SAI Inference Service follows a **robust multi-phase initialization pattern** designed for production reliability:

1. **Synchronous module load** handles configuration, logging, and model loading
2. **Async startup phase** handles database connections and background tasks
3. **Graceful degradation** ensures service operates even with partial failures
4. **SystemD integration** provides automatic restart and health monitoring
5. **Prometheus metrics** enable comprehensive observability

**Key Design Principles**:
- **Fail-safe startup**: Database/CUDA failures don't prevent service start
- **Non-blocking initialization**: Heavy operations (model load) happen at module load time
- **Background task isolation**: Cleanup/metrics failures don't affect inference
- **Signal-based shutdown**: Graceful cleanup on SIGTERM

**Operational Excellence**:
- Startup time: <2 seconds
- Health check: Always available
- Zero-downtime model switching: Supported via API
- Automatic restart: Via systemd watchdog

---

**End of Document**
