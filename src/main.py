"""
SAI Inference Service - FastAPI Application
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Response, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid
import logging
import psutil
import httpx
from datetime import datetime, timezone, timedelta
import json as json_module
from pathlib import Path
from typing import Optional, List
import base64
import asyncio
import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from .config import settings
from .models import (
    InferenceRequest, InferenceResponse,
    BatchInferenceRequest, BatchInferenceResponse,
    HealthCheck, WebhookPayload,
    CameraStats, DetectionRecord, CameraListItem, EscalationEvent,
    AlertSummary, EscalationStats
)
from .inference import inference_engine
from .inference_mosaic import mosaic_inference_engine
from .alert_manager import alert_manager
from .inference_context import InferenceContext
from .database import db_manager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Systemd watchdog support
try:
    from systemd import daemon
    watchdog_enabled = True
    logger.info("SystemD watchdog support enabled")
except ImportError:
    watchdog_enabled = False
    logger.info("SystemD watchdog support not available (systemd-python not installed)")

# Global task state
watchdog_task = None
cleanup_task = None
metrics_task = None

# Prometheus Metrics - Initialize as None first
inference_requests_total = None
detections_total = None
alert_escalations_total = None
alert_levels_total = None
inference_duration_seconds = None
model_inference_duration_seconds = None
active_cameras = None
total_alerts_24h = None
db_query_duration_seconds = None
image_width_pixels = None
image_height_pixels = None
camera_detection_rate = None

if settings.enable_metrics:
    # Request counters
    inference_requests_total = Counter(
        'sai_inference_requests_total',
        'Total number of inference requests',
        ['endpoint', 'status']
    )

    # Detection counters with class and camera labels
    detections_total = Counter(
        'sai_detections_total',
        'Total number of detections',
        ['class_name', 'camera_id', 'endpoint']
    )

    # Alert escalation counters with reason and camera labels
    alert_escalations_total = Counter(
        'sai_alert_escalations_total',
        'Total number of alert escalations',
        ['reason', 'camera_id']
    )

    # Alert level counters
    alert_levels_total = Counter(
        'sai_alert_levels_total',
        'Total alerts by level',
        ['alert_level', 'camera_id', 'endpoint']
    )

    # Inference duration histogram
    inference_duration_seconds = Histogram(
        'sai_inference_duration_seconds',
        'Inference processing time in seconds',
        ['endpoint']
    )

    # Model inference duration
    model_inference_duration_seconds = Histogram(
        'sai_model_inference_duration_seconds',
        'Model inference time in seconds'
    )

    # System metrics
    active_cameras = Gauge(
        'sai_active_cameras',
        'Number of active cameras in last 24 hours'
    )

    total_alerts_24h = Gauge(
        'sai_total_alerts_24h',
        'Total alerts in last 24 hours'
    )

    # Database query performance
    db_query_duration_seconds = Histogram(
        'sai_db_query_duration_seconds',
        'Database query execution time in seconds',
        ['query_type']
    )

    # Image dimension statistics
    image_width_pixels = Histogram(
        'sai_image_width_pixels',
        'Input image width in pixels',
        ['endpoint'],
        buckets=[320, 640, 864, 1024, 1280, 1920, 2560, 3840, 5120, 7680]
    )

    image_height_pixels = Histogram(
        'sai_image_height_pixels',
        'Input image height in pixels',
        ['endpoint'],
        buckets=[240, 480, 640, 768, 1024, 1080, 1440, 2160, 3840, 5760]
    )

    # Per-camera detection rate (detections per request)
    camera_detection_rate = Histogram(
        'sai_camera_detection_rate',
        'Detections per request by camera',
        ['camera_id'],
        buckets=[0, 1, 2, 3, 5, 10, 20, 50, 100]
    )

    logger.info("Prometheus metrics initialized")

# Track inference metrics function (works with or without metrics enabled)
def track_inference_metrics(
    response: InferenceResponse,
    endpoint: str,
    status: str = "success",
    camera_id: Optional[str] = None,
    alert_level: Optional[str] = None
):
    """Track inference metrics for Prometheus"""
    if not settings.enable_metrics or inference_requests_total is None:
        return

    try:
        # Track request
        inference_requests_total.labels(endpoint=endpoint, status=status).inc()

        # Track detections by class
        for detection in response.detections:
            detections_total.labels(
                class_name=detection.class_name,
                camera_id=camera_id or "none",
                endpoint=endpoint
            ).inc()

        # Track alert levels
        if alert_level:
            alert_levels_total.labels(
                alert_level=alert_level,
                camera_id=camera_id or "none",
                endpoint=endpoint
            ).inc()

        # Track inference duration
        inference_duration_seconds.labels(endpoint=endpoint).observe(
            response.processing_time_ms / 1000.0
        )

        # Track image dimensions
        if response.image_size:
            image_width_pixels.labels(endpoint=endpoint).observe(
                response.image_size['width']
            )
            image_height_pixels.labels(endpoint=endpoint).observe(
                response.image_size['height']
            )

        # Track camera detection rate
        if camera_id:
            camera_detection_rate.labels(camera_id=camera_id).observe(
                response.detection_count
            )

    except Exception as e:
        logger.error(f"Failed to track metrics: {e}")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    openapi_url=f"{settings.api_prefix}/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Systemd watchdog functionality
async def watchdog_ping():
    """Send periodic watchdog notifications to systemd"""
    while True:
        try:
            if watchdog_enabled:
                daemon.notify('WATCHDOG=1')
                logger.debug("Sent watchdog ping to systemd")
            await asyncio.sleep(30)  # Send every 30 seconds (less than 60s timeout)
        except Exception as e:
            logger.error(f"Watchdog ping failed: {e}")
            await asyncio.sleep(30)


async def periodic_cleanup():
    """Periodic cleanup of alert data and expired tickets"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            await alert_manager.cleanup_old_data()
            logger.debug("Periodic alert cleanup completed")
        except Exception as e:
            logger.error(f"Periodic cleanup failed: {e}")
            await asyncio.sleep(3600)


async def update_prometheus_gauges():
    """Update Prometheus gauge metrics periodically"""
    while True:
        try:
            await asyncio.sleep(60)  # Update every minute

            if settings.enable_metrics:
                # Update active cameras count
                cameras = await db_manager.get_all_cameras(hours=24)
                active_cameras.set(len(cameras))

                # Update total alerts
                summary = await db_manager.get_alert_summary(hours=24)
                total_alerts_24h.set(summary['total_alerts'])

                logger.debug("Prometheus gauge metrics updated")
        except Exception as e:
            logger.error(f"Prometheus gauge update failed: {e}")
            await asyncio.sleep(60)


@app.on_event("startup")
async def startup_event():
    """Initialize database, cleanup, and watchdog on startup"""
    global watchdog_task, cleanup_task, metrics_task

    # Initialize database for enhanced alert system
    try:
        await db_manager.initialize()
        logger.info("Enhanced alert system database initialized")

        # Start periodic cleanup task
        cleanup_task = asyncio.create_task(periodic_cleanup())
        logger.info("Alert cleanup task started")

        # Start Prometheus gauge update task
        if settings.enable_metrics:
            metrics_task = asyncio.create_task(update_prometheus_gauges())
            logger.info("Prometheus metrics update task started")
    except Exception as e:
        logger.warning(f"Database initialization failed - enhanced alerts disabled: {e}")

    if watchdog_enabled and os.environ.get('WATCHDOG_USEC'):
        # Only start watchdog if systemd expects it
        logger.info("Starting systemd watchdog task")
        watchdog_task = asyncio.create_task(watchdog_ping())
        # Notify systemd that we're ready
        daemon.notify('READY=1')
    else:
        logger.info("SystemD watchdog not required or not available")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up database, tasks, and watchdog on shutdown"""
    global watchdog_task, cleanup_task, metrics_task

    # Stop cleanup task
    if cleanup_task:
        logger.info("Stopping alert cleanup task")
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

    # Stop metrics task
    if metrics_task:
        logger.info("Stopping Prometheus metrics task")
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass

    # Close database connections
    try:
        await db_manager.close()
        logger.info("Database connections closed")
    except Exception as e:
        logger.warning(f"Database shutdown error: {e}")

    if watchdog_task:
        logger.info("Stopping systemd watchdog task")
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass

    if watchdog_enabled:
        daemon.notify('STOPPING=1')


def verify_api_key(api_key: Optional[str] = None) -> bool:
    """Verify API key if configured"""
    if settings.n8n_api_key is None:
        return True
    if api_key is None:
        return False
    import secrets
    return secrets.compare_digest(api_key, settings.n8n_api_key)


def parse_captured_at_from_metadata(cam_metadata: dict) -> datetime:
    """
    Extract and validate captured_at from camera metadata.
    Raises HTTPException(422) if missing or invalid.
    """
    cap_time = cam_metadata.get('environment', {}).get('capture_time_utc')
    if not cap_time:
        raise HTTPException(
            status_code=422,
            detail="metadata.environment.capture_time_utc is required"
        )

    try:
        # Handle Z suffix -> UTC-aware datetime
        if isinstance(cap_time, str) and cap_time.endswith('Z'):
            cap_time = cap_time.replace('Z', '+00:00')
        captured_at = datetime.fromisoformat(cap_time)
        if captured_at.tzinfo is None:
            captured_at = captured_at.replace(tzinfo=timezone.utc)
        captured_at = captured_at.astimezone(timezone.utc)
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid capture_time_utc format: {e}"
        )

    # Guardrails: reject timestamps too far in future or past
    now_utc = datetime.now(timezone.utc)
    if captured_at > now_utc + timedelta(minutes=5):
        raise HTTPException(status_code=422, detail="captured_at is in the future")
    if captured_at < now_utc - timedelta(days=730):
        raise HTTPException(status_code=422, detail="captured_at is too old (>2 years)")

    return captured_at


async def download_image(url: str) -> bytes:
    """Download image from URL"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=30.0)
        response.raise_for_status()
        return response.content


async def send_webhook(url: str, payload: WebhookPayload):
    """Send webhook notification"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload.model_dump(mode="json"),
                timeout=10.0
            )
            response.raise_for_status()
            logger.info(f"Webhook sent successfully to {url}")
    except Exception as e:
        logger.error(f"Failed to send webhook to {url}: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "api_docs": f"{settings.api_prefix}/docs"
    }


@app.get(f"{settings.api_prefix}/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    try:
        model_info = inference_engine.model_manager.get_model_info()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        system_metrics = {
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "memory_available_gb": memory.available / (1024**3)
        }
        
        # Add runtime parameters to ensure source of truth
        runtime_params = {
            "confidence_threshold": settings.confidence_threshold,
            "iou_threshold": settings.iou_threshold,
            "input_size": settings.input_size,
            "device": settings.device,
            "model_dir": str(settings.models_dir),
            "default_model": settings.default_model
        }
        
        # Update model info with actual runtime values if loaded
        if model_info:
            model_info.input_size = settings.input_size
            model_info.confidence_threshold = settings.confidence_threshold
            model_info.iou_threshold = settings.iou_threshold
        
        return HealthCheck(
            status="healthy",
            timestamp=datetime.utcnow(),
            version=settings.app_version,
            is_model_loaded=model_info is not None,
            loaded_model_info=model_info,
            system_metrics=system_metrics,
            runtime_parameters=runtime_params
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get(f"{settings.api_prefix}/models")
async def list_models():
    """List available models with smart discovery"""
    loaded_models = inference_engine.model_manager.list_models()
    model_info = [
        inference_engine.model_manager.get_model_info(name)
        for name in loaded_models
    ]
    
    # Use smart discovery for available models
    discovered_models = inference_engine.model_manager.discover_models()
    available_files = [
        model for model in discovered_models 
        if model not in loaded_models
    ]
    
    # Get recommended model
    recommended_model = inference_engine.model_manager.get_best_available_model()
    
    return {
        "loaded_models": model_info,
        "available_files": available_files,
        "current_model": inference_engine.model_manager.current_model_name,
        "recommended_model": recommended_model,
        "configured_default": settings.default_model,
        "discovery_info": {
            "models_dir": str(settings.models_dir),
            "total_discovered": len(discovered_models),
            "supported_extensions": [".pt", ".pth", ".onnx", ".engine"]
        }
    }


@app.post(f"{settings.api_prefix}/models/load")
async def load_model(model_name: str):
    """Load a new model"""
    success = inference_engine.model_manager.load_model(model_name)
    if success:
        return {"message": f"Model {model_name} loaded successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to load model {model_name}")


@app.post(f"{settings.api_prefix}/models/switch")
async def switch_model(model_name: str):
    """Switch to a different loaded model"""
    success = inference_engine.model_manager.set_current_model(model_name)
    if success:
        return {"message": f"Switched to model {model_name}"}
    else:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not loaded")


@app.post(f"{settings.api_prefix}/infer", response_model=InferenceResponse)
async def infer(
    file: UploadFile = File(...),
    # Core Detection Parameters
    confidence_threshold: Optional[float] = Form(None),
    iou_threshold: Optional[float] = Form(None),
    max_detections: Optional[int] = Form(None),
    # Enhanced Alert System
    camera_id: Optional[str] = Form(None, description="Camera identifier for enhanced temporal alert tracking"),
    # High-Value YOLO Parameters
    detection_classes: Optional[str] = Form(None, description="JSON array: [0] for smoke, [1] for fire, [0,1] for both"),
    half_precision: Optional[str] = Form("false", description="true/false"),
    test_time_augmentation: Optional[str] = Form("false", description="true/false"),
    class_agnostic_nms: Optional[str] = Form("false", description="true/false"),
    # Annotation Control
    return_image: Optional[str] = Form("false", description="true/false"),
    show_labels: Optional[str] = Form("true", description="true/false"),
    show_confidence: Optional[str] = Form("true", description="true/false"),
    line_width: Optional[int] = Form(None),
    # Metadata (contains capture_time_utc from camera node)
    metadata_file: Optional[UploadFile] = File(None),
    # Processing Options
    webhook_url: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    """Run inference on binary image data (n8n compatible)"""
    request_id = str(uuid.uuid4())

    # Parse metadata file to extract captured_at
    cam_metadata = {}
    if metadata_file:
        raw = await metadata_file.read()
        try:
            cam_metadata = json_module.loads(raw)
        except (json_module.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=422, detail=f"Invalid metadata JSON: {e}")
    captured_at = parse_captured_at_from_metadata(cam_metadata)
    
    # Check file extension
    file_ext = Path(file.filename or "image.jpg").suffix.lower()
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not supported"
        )
    
    # Check file size
    contents = await file.read()
    if len(contents) > settings.max_upload_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_size / (1024*1024):.1f}MB"
        )
    
    try:
        # Log incoming camera_id for debugging n8n integration
        if camera_id:
            logger.info(f"[n8n] Request {request_id} - camera_id received: '{camera_id}'")
        else:
            logger.debug(f"[n8n] Request {request_id} - no camera_id provided")

        # Parse form data parameters
        def parse_bool(value: Optional[str], default: bool = False) -> bool:
            if value is None:
                return default
            return value.lower() in ("true", "1", "yes", "on")
        
        def parse_json_array(value: Optional[str]) -> Optional[List[int]]:
            if value is None:
                return None
            try:
                import json
                parsed = json.loads(value)
                if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
                    return parsed
                else:
                    raise ValueError("Must be array of integers")
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid detection_classes format: {e}")
        
        # Convert form data to typed parameters
        return_annotated_image = parse_bool(return_image)
        parsed_detection_classes = parse_json_array(detection_classes)
        parsed_half_precision = parse_bool(half_precision)
        parsed_tta = parse_bool(test_time_augmentation)
        parsed_agnostic_nms = parse_bool(class_agnostic_nms)
        parsed_show_labels = parse_bool(show_labels, True)
        parsed_show_confidence = parse_bool(show_confidence, True)
        
        # Run inference with enhanced parameters
        response = await inference_engine.infer(
            image_data=contents,  # Raw bytes directly
            request_id=request_id,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            return_annotated=return_annotated_image,
            # High-Value YOLO Parameters
            detection_classes=parsed_detection_classes,
            half_precision=parsed_half_precision,
            test_time_augmentation=parsed_tta,
            class_agnostic_nms=parsed_agnostic_nms,
            # Annotation Control
            show_labels=parsed_show_labels,
            show_confidence=parsed_show_confidence,
            line_width=line_width,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "source": "binary_upload"
            }
        )

        # Build inference context with captured_at for DB storage
        context = InferenceContext(
            request_id=request_id,
            camera_id=camera_id or "unknown",
            detections=response.detections,
            captured_at=captured_at,
            detection_count=response.detection_count,
            max_confidence=max(d.confidence for d in response.detections) if response.detections else 0.0,
            processing_time_ms=response.processing_time_ms,
            model_inference_time_ms=response.model_inference_time_ms,
            image_width=response.image_size.get('width') if response.image_size else None,
            image_height=response.image_size.get('height') if response.image_size else None,
            source="n8n",
            metadata=cam_metadata,
        )

        # Determine alert level via alert_manager (temporal analysis + DB logging)
        computed_alert_level = await alert_manager.determine_alert_level(
            response.detections, camera_id, context=context
        )

        # Send webhook if requested
        if webhook_url:
            webhook_payload = WebhookPayload(
                event_type="detection",
                timestamp=datetime.utcnow(),
                source="sai-inference",
                data=response
            )
            webhook_payload.alert_level = computed_alert_level
            background_tasks.add_task(send_webhook, webhook_url, webhook_payload)

        # Track metrics
        track_inference_metrics(
            response, endpoint="/api/v1/infer", status="success",
            camera_id=camera_id, alert_level=computed_alert_level
        )

        return response

    except Exception as e:
        logger.error(f"Binary inference failed: {e}")
        if settings.enable_metrics and inference_requests_total:
            inference_requests_total.labels(endpoint="/api/v1/infer", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{settings.api_prefix}/infer/base64", response_model=InferenceResponse)
async def infer_base64(request: InferenceRequest, background_tasks: BackgroundTasks):
    """Run inference on base64 encoded image (legacy/secondary channel)"""
    request_id = str(uuid.uuid4())
    
    try:
        # Get image data
        if request.image:
            image_data = request.image
        elif request.image_url:
            image_bytes = await download_image(request.image_url)
            image_data = base64.b64encode(image_bytes).decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Extract captured_at from metadata (mandatory)
        req_metadata = request.metadata or {}
        req_camera_id = req_metadata.get("camera_id")
        captured_at = parse_captured_at_from_metadata(req_metadata)
        response = await inference_engine.infer(
            image_data=image_data,
            request_id=request_id,
            confidence_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold,
            max_detections=request.max_detections,
            return_annotated=request.return_image,
            # High-Value YOLO Parameters
            detection_classes=request.detection_classes,
            half_precision=request.half_precision,
            test_time_augmentation=request.test_time_augmentation,
            class_agnostic_nms=request.class_agnostic_nms,
            # Annotation Control
            show_labels=request.show_labels,
            show_confidence=request.show_confidence,
            line_width=request.line_width,
            metadata={
                **request.metadata,
                "source": "base64_json",
                "enhanced_features": {
                    "detection_classes": request.detection_classes,
                    "half_precision": request.half_precision,
                    "test_time_augmentation": request.test_time_augmentation,
                    "class_agnostic_nms": request.class_agnostic_nms
                }
            }
        )

        # Build inference context with captured_at
        context = InferenceContext(
            request_id=request_id,
            camera_id=req_camera_id or "unknown",
            detections=response.detections,
            captured_at=captured_at,
            detection_count=response.detection_count,
            max_confidence=max(d.confidence for d in response.detections) if response.detections else 0.0,
            processing_time_ms=response.processing_time_ms,
            model_inference_time_ms=response.model_inference_time_ms,
            source="base64_json",
            metadata=req_metadata,
        )

        # Determine alert level via alert_manager
        computed_alert_level = await alert_manager.determine_alert_level(
            response.detections, req_camera_id, context=context
        )

        # Send webhook if requested
        if request.webhook_url:
            webhook_payload = WebhookPayload(
                event_type="detection",
                timestamp=datetime.utcnow(),
                source="sai-inference",
                data=response
            )
            webhook_payload.alert_level = computed_alert_level
            background_tasks.add_task(send_webhook, request.webhook_url, webhook_payload)

        # Track metrics
        track_inference_metrics(
            response, endpoint="/api/v1/infer/base64", status="success",
            camera_id=req_camera_id, alert_level=computed_alert_level
        )

        return response

    except Exception as e:
        logger.error(f"Base64 inference failed: {e}")
        if settings.enable_metrics and inference_requests_total:
            inference_requests_total.labels(endpoint="/api/v1/infer/base64", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{settings.api_prefix}/infer/mosaic", response_model=InferenceResponse)
async def infer_mosaic(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Form(None),
    iou_threshold: Optional[float] = Form(None),
    camera_id: Optional[str] = Form(None, description="Camera identifier for enhanced temporal alert tracking"),
    metadata_file: Optional[UploadFile] = File(None),
    return_image: Optional[str] = Form("false"),
    webhook_url: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    """Run mosaic inference on large images using 640x640 overlapping crops"""
    request_id = str(uuid.uuid4())

    # Parse metadata file to extract captured_at
    cam_metadata = {}
    if metadata_file:
        raw = await metadata_file.read()
        try:
            cam_metadata = json_module.loads(raw)
        except (json_module.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=422, detail=f"Invalid metadata JSON: {e}")
    captured_at = parse_captured_at_from_metadata(cam_metadata)

    # Check file extension
    file_ext = Path(file.filename or "image.jpg").suffix.lower()
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not supported"
        )
    
    # Check file size
    contents = await file.read()
    if len(contents) > settings.max_upload_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_size / (1024*1024):.1f}MB"
        )
    
    try:
        # Convert form data string to boolean
        return_annotated_image = bool(return_image and return_image.lower() in ("true", "1", "yes", "on"))
        
        # Run mosaic inference using 640x640 crops
        response = await mosaic_inference_engine.infer_mosaic(
            image_data=contents,  # Raw bytes directly
            request_id=request_id,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            return_annotated=return_annotated_image,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "source": "mosaic_inference",
                "endpoint": "/api/v1/infer/mosaic"
            }
        )
        
        # Build inference context with captured_at
        context = InferenceContext(
            request_id=request_id,
            camera_id=camera_id or "unknown",
            detections=response.detections,
            captured_at=captured_at,
            detection_count=response.detection_count,
            max_confidence=max(d.confidence for d in response.detections) if response.detections else 0.0,
            processing_time_ms=response.processing_time_ms,
            source="mosaic",
            metadata=cam_metadata,
        )

        # Determine alert level via alert_manager
        computed_alert_level = await alert_manager.determine_alert_level(
            response.detections, camera_id, context=context
        )

        # Send webhook if requested
        if webhook_url:
            webhook_payload = WebhookPayload(
                event_type="detection",
                timestamp=datetime.utcnow(),
                source="sai-inference-mosaic",
                data=response
            )
            webhook_payload.alert_level = computed_alert_level
            background_tasks.add_task(send_webhook, webhook_url, webhook_payload)

        # Track metrics
        track_inference_metrics(
            response, endpoint="/api/v1/infer/mosaic", status="success",
            camera_id=camera_id, alert_level=computed_alert_level
        )

        return response

    except Exception as e:
        logger.error(f"Mosaic inference failed: {e}")
        if settings.enable_metrics and inference_requests_total:
            inference_requests_total.labels(endpoint="/api/v1/infer/mosaic", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{settings.api_prefix}/infer/batch", response_model=BatchInferenceResponse)
async def infer_batch(request: BatchInferenceRequest):
    """Run inference on multiple images"""
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()

    try:
        # Process images
        results = await inference_engine.infer_batch(
            images=request.images,
            request_id=request_id,
            confidence_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold,
            max_detections=request.max_detections_per_image,
            return_annotated=request.return_images,
            metadata=request.metadata
        )

        # Calculate summary
        total_detections = sum(r.detection_count for r in results)
        total_fire = sum(1 for r in results if r.has_fire)
        total_smoke = sum(1 for r in results if r.has_smoke)

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return BatchInferenceResponse(
            request_id=request_id,
            timestamp=start_time,
            total_processing_time_ms=processing_time,
            results=results,
            total_detections=total_detections,
            summary={
                "total_images": len(results),
                "images_with_fire": total_fire,
                "images_with_smoke": total_smoke,
                "average_detections_per_image": total_detections / len(results) if results else 0
            },
            metadata=request.metadata
        )

    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Camera Analytics API
# ============================================================================

@app.get(f"{settings.api_prefix}/cameras", response_model=List[CameraListItem])
async def list_cameras(hours: int = 24):
    """List all cameras with recent activity"""
    try:
        cameras = await db_manager.get_all_cameras(hours=hours)
        return cameras
    except Exception as e:
        logger.error(f"Failed to list cameras: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(f"{settings.api_prefix}/cameras/{{camera_id}}/stats", response_model=CameraStats)
async def get_camera_statistics(camera_id: str, hours: int = 24):
    """Get detection statistics for a specific camera"""
    try:
        stats = await db_manager.get_camera_stats(camera_id=camera_id, hours=hours)
        return stats
    except Exception as e:
        logger.error(f"Failed to get camera stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(f"{settings.api_prefix}/cameras/{{camera_id}}/detections", response_model=List[DetectionRecord])
async def get_camera_detections(
    camera_id: str,
    minutes: int = 180,
    min_confidence: Optional[float] = None
):
    """Get recent detections for a specific camera"""
    try:
        detections = await db_manager.get_detections_with_metadata(
            camera_id=camera_id,
            minutes=minutes,
            min_confidence=min_confidence
        )
        return detections
    except Exception as e:
        logger.error(f"Failed to get camera detections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(f"{settings.api_prefix}/cameras/{{camera_id}}/escalations", response_model=List[EscalationEvent])
async def get_camera_escalations(camera_id: str, hours: int = 24):
    """Get escalation events for a specific camera"""
    try:
        escalations = await db_manager.get_escalation_events(camera_id=camera_id, hours=hours)
        return escalations
    except Exception as e:
        logger.error(f"Failed to get camera escalations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Alert History API
# ============================================================================

@app.get(f"{settings.api_prefix}/alerts/recent", response_model=List[DetectionRecord])
async def get_recent_alerts(limit: int = 100, camera_id: Optional[str] = None):
    """Get recent alerts across all cameras or for a specific camera"""
    try:
        alerts = await db_manager.get_recent_alerts(limit=limit, camera_id=camera_id)
        return alerts
    except Exception as e:
        logger.error(f"Failed to get recent alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(f"{settings.api_prefix}/alerts/summary", response_model=AlertSummary)
async def get_alert_summary(hours: int = 24, camera_id: Optional[str] = None):
    """Get alert summary statistics"""
    try:
        summary = await db_manager.get_alert_summary(hours=hours, camera_id=camera_id)
        return summary
    except Exception as e:
        logger.error(f"Failed to get alert summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(f"{settings.api_prefix}/alerts/escalation-stats", response_model=EscalationStats)
async def get_escalation_statistics(hours: int = 24, camera_id: Optional[str] = None):
    """Get escalation statistics"""
    try:
        stats = await db_manager.get_escalation_stats(hours=hours, camera_id=camera_id)
        return stats
    except Exception as e:
        logger.error(f"Failed to get escalation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Prometheus Metrics Endpoint
# ============================================================================

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if not settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics disabled")

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)







def run_server():
    """Run the FastAPI server"""
    logger.info(f"Starting {settings.app_name} on {settings.host}:{settings.port}")
    logger.info(f"Device: {inference_engine.model_manager.device}")
    
    # Log runtime parameters for source of truth verification
    logger.info("=" * 60)
    logger.info("RUNTIME PARAMETERS (Source of Truth):")
    logger.info(f"  Confidence Threshold: {settings.confidence_threshold}")
    logger.info(f"  IOU Threshold: {settings.iou_threshold}")
    logger.info(f"  Input Size: {settings.input_size}x{settings.input_size}")
    logger.info(f"  Model Directory: {settings.models_dir}")
    logger.info(f"  Default Model: {settings.default_model}")
    logger.info(f"  Device: {settings.device}")
    logger.info("=" * 60)
    
    logger.info(f"API docs available at: http://{settings.host}:{settings.port}{settings.api_prefix}/docs")
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=False
    )


if __name__ == "__main__":
    run_server()