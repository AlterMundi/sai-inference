# SAI Inference Service - Visual Initialization Diagram

## Complete Initialization Flow (ASCII Diagram)

```
═══════════════════════════════════════════════════════════════════════════════
                        SAI INFERENCE SERVICE STARTUP
═══════════════════════════════════════════════════════════════════════════════

┌───────────────────────────────────────────────────────────────────────────┐
│                    ENTRY POINT: python run.py                             │
│                    OR: uvicorn src.main:app                               │
└───────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
╔═══════════════════════════════════════════════════════════════════════════╗
║ PHASE 1: MODULE IMPORT & CONFIGURATION (Synchronous, ~500ms)             ║
╚═══════════════════════════════════════════════════════════════════════════╝
                                      │
                    ┌─────────────────┴─────────────────┐
                    │   from src.main import app        │
                    └─────────────────┬─────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌────────────────┐        ┌────────────────────┐      ┌─────────────────────┐
│ src/config.py  │        │  src/inference.py  │      │    src/database.py  │
│                │        │                    │      │                     │
│ • load_dotenv()│        │ • ModelManager()   │      │ • DatabaseManager() │
│ • Settings()   │        │   - device detect  │      │   (no conn yet)     │
│   validation   │        │   - discover models│      │                     │
└────────┬───────┘        │   - load YOLO      │      └─────────────────────┘
         │                │   - to(device)     │
         │                └──────────┬─────────┘
         │                           │
         │                           │
         ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         GLOBAL STATE CREATED                            │
│  • settings: Settings object (from .env)                                │
│  • logger: Configured logging                                           │
│  • watchdog_enabled: bool (systemd detection)                           │
│  • inference_engine: InferenceEngine (with loaded YOLO model)           │
│  • db_manager: DatabaseManager (pool=None)                              │
│  • alert_manager: AlertManager (db_initialized=False)                   │
│  • Prometheus metrics: Counters/Histograms/Gauges (if enabled)          │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │  FastAPI app = FastAPI(...)       │
                    │  + CORS middleware                │
                    │  + Route registration             │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
╔═══════════════════════════════════════════════════════════════════════════╗
║ PHASE 2: UVICORN SERVER START                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
                                      │
                    ┌─────────────────┴─────────────────┐
                    │  uvicorn.run(app, ...)            │
                    │  • Bind to host:port              │
                    │  • Create event loop              │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
╔═══════════════════════════════════════════════════════════════════════════╗
║ PHASE 3: FASTAPI STARTUP EVENT (Async, ~100-500ms)                       ║
╚═══════════════════════════════════════════════════════════════════════════╝
                                      │
                    ┌─────────────────┴─────────────────┐
                    │  @app.on_event("startup")         │
                    └─────────────────┬─────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│ DATABASE INIT   │      │  BACKGROUND TASKS    │      │  SYSTEMD NOTIFY  │
│                 │      │                      │      │                  │
│ • asyncpg.pool  │      │ • periodic_cleanup() │      │ • daemon.notify  │
│   create_pool() │      │   (every 1h)         │      │   ('READY=1')    │
│ • create_tables │      │ • update_prometheus  │      │ • watchdog_ping  │
│   (if needed)   │      │   _gauges() (1m)     │      │   (every 30s)    │
│                 │      │ • watchdog_ping()    │      │                  │
│ SUCCESS ✓       │      │   (every 30s)        │      │ SUCCESS ✓        │
│ or FALLBACK ⚠   │      │                      │      │ or SKIP ⚠        │
└────────┬────────┘      └──────────┬───────────┘      └──────────────────┘
         │                          │
         └──────────────┬───────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                  BACKGROUND TASKS SPAWNED (Asyncio)                       │
│                                                                           │
│  Task 1: cleanup_task = asyncio.create_task(periodic_cleanup())          │
│  Task 2: metrics_task = asyncio.create_task(update_prometheus_gauges())  │
│  Task 3: watchdog_task = asyncio.create_task(watchdog_ping())            │
│                                                                           │
│  All tasks: Non-blocking, self-healing on errors                         │
└───────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
╔═══════════════════════════════════════════════════════════════════════════╗
║ PHASE 4: STABLE OPERATION (Event Loop Running)                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
                                      │
                    ┌─────────────────┴─────────────────┐
                    │  Service Ready: Accept Requests   │
                    └─────────────────┬─────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│ HTTP ENDPOINTS  │      │  BACKGROUND LOOP     │      │  MONITORING      │
│                 │      │                      │      │                  │
│ • /api/v1/infer │      │ Cleanup: 1h interval │      │ • /api/v1/health │
│ • /api/v1/      │      │ Metrics: 1m interval │      │ • /metrics       │
│   infer/base64  │      │ Watchdog: 30s ping   │      │   (Prometheus)   │
│ • /api/v1/      │      │                      │      │ • SystemD status │
│   infer/mosaic  │      │ All self-healing     │      │   (watchdog OK)  │
│ • /api/v1/models│      │                      │      │                  │
│ • /api/v1/      │      │                      │      │                  │
│   cameras       │      │                      │      │                  │
│ • /api/v1/alerts│      │                      │      │                  │
└─────────────────┘      └──────────────────────┘      └──────────────────┘
                                      │
                                      │
                    ┌─────────────────┴─────────────────┐
                    │     Request Handling Flow         │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                       INFERENCE REQUEST LIFECYCLE                         │
│                                                                           │
│  1. POST /api/v1/infer (image + params)                                  │
│      ↓                                                                    │
│  2. Validate file size/extension                                         │
│      ↓                                                                    │
│  3. Decode image (base64 or binary)                                      │
│      ↓                                                                    │
│  4. YOLO inference (inference_engine)                                    │
│      • Preprocess (letterbox, normalize)                                 │
│      • Model forward pass                                                │
│      • NMS (non-max suppression)                                         │
│      • Parse detections (bbox + confidence)                              │
│      ↓                                                                    │
│  5. Alert determination (alert_manager)                                  │
│      • Base level: confidence thresholds                                 │
│      • Enhanced mode: temporal escalation (if camera_id)                 │
│      ↓                                                                    │
│  6. Database logging (if camera_id)                                      │
│      • Store detection record                                            │
│      • Check temporal patterns (30m/3h windows)                          │
│      • Calculate escalation reason                                       │
│      ↓                                                                    │
│  7. Response generation                                                   │
│      • Detection array (class, confidence, bbox)                         │
│      • Alert level (none/low/high/critical)                              │
│      • Annotated image (if requested)                                    │
│      • Metrics (processing_time_ms, confidence_scores)                   │
│      ↓                                                                    │
│  8. Prometheus metrics tracking                                          │
│      • Increment request counter                                         │
│      • Record latency histogram                                          │
│      • Update detection counters                                         │
│      ↓                                                                    │
│  9. HTTP 200 JSON response                                               │
│                                                                           │
│  Typical Latency: 80-100ms (CPU), 40-60ms (GPU)                          │
└───────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │
                    ┌─────────────────┴─────────────────┐
                    │  SIGTERM received (shutdown)      │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
╔═══════════════════════════════════════════════════════════════════════════╗
║ PHASE 5: GRACEFUL SHUTDOWN (1-5 seconds)                                 ║
╚═══════════════════════════════════════════════════════════════════════════╝
                                      │
                    ┌─────────────────┴─────────────────┐
                    │  @app.on_event("shutdown")        │
                    └─────────────────┬─────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│ CANCEL TASKS    │      │  CLOSE DATABASE      │      │  NOTIFY SYSTEMD  │
│                 │      │                      │      │                  │
│ • cleanup_task  │      │ • await pool.close() │      │ • daemon.notify  │
│   .cancel()     │      │ • Close all conns    │      │   ('STOPPING=1') │
│ • metrics_task  │      │                      │      │                  │
│   .cancel()     │      │                      │      │                  │
│ • watchdog_task │      │                      │      │                  │
│   .cancel()     │      │                      │      │                  │
└────────┬────────┘      └──────────┬───────────┘      └──────────────────┘
         │                          │
         └──────────────┬───────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         PROCESS EXIT (code 0)                             │
└───────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                              FAILURE MODES
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│ Failure: No YOLO model in models/ directory                            │
│ Behavior: Service starts, health returns is_model_loaded=false         │
│ Recovery: Add model file, call /api/v1/models/load (no restart)        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Failure: Database connection refused                                    │
│ Behavior: Service starts, enhanced alerts disabled (basic mode only)   │
│ Recovery: Fix SAI_DATABASE_URL, restart service                        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Failure: CUDA requested but unavailable                                 │
│ Behavior: Falls back to CPU, logs warning, inference slower            │
│ Recovery: Install CUDA drivers, set SAI_DEVICE=cuda:0, restart         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Failure: Watchdog timeout (no ping for 60s)                             │
│ Behavior: SystemD sends SIGTERM, service restarts automatically        │
│ Recovery: Automatic via systemd Restart=always                          │
└─────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                          COMPONENT DEPENDENCIES
═══════════════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────────┐
│                         CRITICAL (Required)                              │
│  ✓ Python 3.11+                                                          │
│  ✓ FastAPI + Uvicorn                                                     │
│  ✓ Ultralytics YOLO                                                      │
│  ✓ PyTorch (CPU or CUDA)                                                 │
│  ✓ Pydantic Settings                                                     │
│  ✓ YOLO model file (.pt format)                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                    OPTIONAL (Graceful Degradation)                       │
│  ⚠ PostgreSQL database → Falls back to basic alert mode                 │
│  ⚠ systemd-python → Watchdog disabled, service still works              │
│  ⚠ CUDA/GPU → Falls back to CPU inference                               │
│  ⚠ Prometheus endpoint → Metrics disabled, inference unaffected         │
└──────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                              TIMING BREAKDOWN
═══════════════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Module Import & Config                              ~500ms     │
│   • Load .env                                                  10ms      │
│   • Settings validation                                        20ms      │
│   • Logging config                                             5ms       │
│   • Prometheus metrics init                                    50ms      │
│   • ModelManager device detect                                 10ms      │
│   • YOLO model load + to(device)                               400ms     │
│   • FastAPI app creation                                       5ms       │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Uvicorn Server Start                                ~50ms      │
│   • Bind socket                                                20ms      │
│   • Event loop creation                                        30ms      │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ Phase 3: FastAPI Startup Event                               ~200ms     │
│   • Database pool creation                                     100ms     │
│   • Table creation (if needed)                                 50ms      │
│   • Background task spawn                                      10ms      │
│   • SystemD notify                                             5ms       │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ Total Cold Start Time:                                       ~750ms     │
│ (Module load + Server start + Startup event)                            │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Stable Operation (per request)                      ~80ms      │
│   • File validation                                            5ms       │
│   • Image decode                                               10ms      │
│   • YOLO inference                                             50ms      │
│   • Alert determination                                        5ms       │
│   • Database logging                                           5ms       │
│   • Response serialization                                     5ms       │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ Phase 5: Graceful Shutdown                                    ~1s       │
│   • Task cancellation                                          100ms     │
│   • Database pool close                                        500ms     │
│   • SystemD notify                                             5ms       │
│   • Process cleanup                                            400ms     │
└──────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                         END OF INITIALIZATION DIAGRAM
═══════════════════════════════════════════════════════════════════════════════
```

## Key Observations

### 1. Layered Initialization
The service uses a **3-phase initialization pattern** to ensure robust startup:
- **Phase 1 (Synchronous)**: Configuration, logging, model loading - happens at module import time
- **Phase 2 (Server Start)**: Uvicorn binds to socket and creates event loop
- **Phase 3 (Async)**: Database connections, background tasks - happens at FastAPI startup event

### 2. Non-Blocking Failures
Critical design principle: **Failures in optional components don't prevent startup**
- Database unreachable → Service continues with basic alert mode
- CUDA unavailable → Falls back to CPU inference
- SystemD not available → Watchdog disabled, service works normally

### 3. Background Task Isolation
All background tasks run independently and self-heal:
- **Cleanup task**: Errors logged, retries after 1 hour
- **Metrics task**: Errors logged, retries after 1 minute
- **Watchdog task**: Errors logged, retries after 30 seconds

None of these failures affect inference request handling.

### 4. SystemD Integration
The service properly integrates with systemd lifecycle:
- **READY=1**: Notifies systemd when startup complete
- **WATCHDOG=1**: Sends keepalive every 30s (timeout: 60s)
- **STOPPING=1**: Notifies systemd during graceful shutdown

### 5. Graceful Degradation Hierarchy
```
Full Operation (All systems go)
    ↓ Database fails
Enhanced Alert Mode Disabled (Basic mode only)
    ↓ GPU fails
CPU Inference (Slower but functional)
    ↓ Model fails
Service Starts but Rejects Requests
    ↓ Port in use
Service Fails to Start (systemd restarts)
```

### 6. Health Check Philosophy
The `/api/v1/health` endpoint returns HTTP 200 even with degraded components:
- `is_model_loaded: false` → Warning state, but service responds
- Database disabled → Not reflected in health check (assumed transient)

This prevents false-positive alerts in monitoring systems.

---

**See also**: [SERVICE_INITIALIZATION_FLOW.md](SERVICE_INITIALIZATION_FLOW.md) for detailed textual analysis.
