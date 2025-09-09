"""
SAI Inference Service - FastAPI Application
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import uuid
import logging
import psutil
import aiofiles
import httpx
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import base64
import io
import asyncio
import os

from .config import settings
from .models import (
    InferenceRequest, InferenceResponse,
    BatchInferenceRequest, BatchInferenceResponse,
    HealthCheck, ModelInfo, ErrorResponse,
    WebhookPayload
)
from .inference import inference_engine

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

# Global watchdog state
watchdog_task = None

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


@app.on_event("startup")
async def startup_event():
    """Initialize watchdog on startup"""
    global watchdog_task
    
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
    """Clean up watchdog on shutdown"""
    global watchdog_task
    
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
    return api_key == settings.n8n_api_key


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
            "memory_available_gb": memory.available / (1024**3),
            "cached_results": 0  # Cache removed to fix identical output bug
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
    confidence_threshold: Optional[float] = Form(None),
    iou_threshold: Optional[float] = Form(None),
    return_image: Optional[str] = Form("false"),
    webhook_url: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    """Run inference on binary image data (n8n compatible)"""
    request_id = str(uuid.uuid4())
    
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
        
        # Pass binary data directly to inference engine (optimal path)
        # No base64 conversion needed - direct bytes → PIL Image → YOLO
        
        # Run inference
        response = await inference_engine.infer(
            image_data=contents,  # Raw bytes directly
            request_id=request_id,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            return_annotated=return_annotated_image,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "source": "binary_upload"
            }
        )
        
        # Send webhook if requested
        if webhook_url:
            webhook_payload = WebhookPayload(
                event_type="detection",
                timestamp=datetime.utcnow(),
                source="sai-inference",
                data=response
            )
            webhook_payload.alert_level = webhook_payload.determine_alert_level()
            background_tasks.add_task(send_webhook, webhook_url, webhook_payload)
        
        return response
        
    except Exception as e:
        logger.error(f"Binary inference failed: {e}")
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
        
        # Run inference
        response = await inference_engine.infer(
            image_data=image_data,
            request_id=request_id,
            confidence_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold,
            max_detections=request.max_detections,
            return_annotated=request.return_image,
            metadata={
                **request.metadata,
                "source": "base64_json"
            }
        )
        
        # Send webhook if requested
        if request.webhook_url:
            webhook_payload = WebhookPayload(
                event_type="detection",
                timestamp=datetime.utcnow(),
                source="sai-inference",
                data=response
            )
            webhook_payload.alert_level = webhook_payload.determine_alert_level()
            background_tasks.add_task(send_webhook, request.webhook_url, webhook_payload)
        
        return response
        
    except Exception as e:
        logger.error(f"Base64 inference failed: {e}")
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


# n8n Webhook endpoints
@app.post(settings.n8n_webhook_path)
async def n8n_webhook_binary(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Form(None),
    iou_threshold: Optional[float] = Form(None),
    return_image: Optional[str] = Form("false"),
    workflow_id: Optional[str] = Form(None),
    execution_id: Optional[str] = Form(None),
    callback_url: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
    api_key: Optional[str] = Form(None)
):
    """n8n webhook endpoint for binary image processing (primary)"""
    
    # Verify API key
    if not verify_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    request_id = str(uuid.uuid4())
    
    try:
        # Read binary data
        contents = await file.read()
        
        # Check file size
        if len(contents) > settings.max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {settings.max_upload_size / (1024*1024):.1f}MB"
            )
        
        # Convert form data string to boolean
        return_annotated_image = bool(return_image and return_image.lower() in ("true", "1", "yes", "on"))
        
        # Pass binary data directly (optimal n8n path)
        # No base64 conversion - direct bytes → PIL Image → YOLO
        
        # Run inference
        response = await inference_engine.infer(
            image_data=contents,  # Raw bytes directly
            request_id=request_id,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            return_annotated=return_annotated_image,
            metadata={
                "source": "n8n_webhook_binary",
                "filename": file.filename,
                "content_type": file.content_type,
                "workflow_id": workflow_id,
                "execution_id": execution_id
            }
        )
        
        # Create webhook payload
        webhook_payload = WebhookPayload(
            event_type="detection",
            timestamp=datetime.utcnow(),
            source="sai-inference",
            data=response
        )
        webhook_payload.alert_level = webhook_payload.determine_alert_level()
        
        # Send to callback URL if provided
        if callback_url:
            background_tasks.add_task(send_webhook, callback_url, webhook_payload)
        
        # Return response in n8n-friendly format
        return {
            "success": True,
            "request_id": request_id,
            "detections": len(response.detections),
            "has_fire": response.has_fire,
            "has_smoke": response.has_smoke,
            "alert_level": webhook_payload.alert_level,
            "processing_time_ms": response.processing_time_ms,
            "data": response.model_dump(mode="json")
        }
        
    except Exception as e:
        logger.error(f"n8n binary webhook processing failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "request_id": request_id
        }


@app.post(f"{settings.n8n_webhook_path}/json")
async def n8n_webhook_json(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = None
):
    """n8n webhook endpoint for JSON/base64 image processing (legacy)"""
    
    # Verify API key
    if not verify_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Extract image from n8n payload
        # n8n can send data in various formats
        image_data = None
        
        # Check common n8n data structures
        if "image" in request:
            image_data = request["image"]
        elif "binary" in request and "data" in request["binary"]:
            # n8n binary data format
            binary_data = request["binary"]["data"]
            if isinstance(binary_data, dict) and "data" in binary_data:
                image_data = binary_data["data"]
        elif "json" in request and "image" in request["json"]:
            image_data = request["json"]["image"]
        
        if not image_data:
            raise ValueError("No image data found in request")
        
        # Create inference request
        request_id = str(uuid.uuid4())
        
        # Run inference
        response = await inference_engine.infer(
            image_data=image_data,
            request_id=request_id,
            confidence_threshold=request.get("confidence_threshold"),
            iou_threshold=request.get("iou_threshold"),
            return_annotated=request.get("return_image", False),
            metadata={
                "source": "n8n_webhook_json",
                "workflow_id": request.get("workflow_id"),
                "execution_id": request.get("execution_id")
            }
        )
        
        # Create webhook payload
        webhook_payload = WebhookPayload(
            event_type="detection",
            timestamp=datetime.utcnow(),
            source="sai-inference",
            data=response
        )
        webhook_payload.alert_level = webhook_payload.determine_alert_level()
        
        # Send to callback URL if provided
        callback_url = request.get("callback_url")
        if callback_url:
            background_tasks.add_task(send_webhook, callback_url, webhook_payload)
        
        # Return response in n8n-friendly format
        return {
            "success": True,
            "request_id": request_id,
            "detections": len(response.detections),
            "has_fire": response.has_fire,
            "has_smoke": response.has_smoke,
            "alert_level": webhook_payload.alert_level,
            "data": response.model_dump(mode="json")
        }
        
    except Exception as e:
        logger.error(f"n8n JSON webhook processing failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# WebSocket endpoint for real-time inference
@app.websocket(f"{settings.api_prefix}/ws")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time inference"""
    await websocket.accept()
    
    try:
        while True:
            # Receive image data
            data = await websocket.receive_json()
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            
            # Process inference request
            request_id = str(uuid.uuid4())
            
            try:
                response = await inference_engine.infer(
                    image_data=data.get("image"),
                    request_id=request_id,
                    confidence_threshold=data.get("confidence_threshold"),
                    iou_threshold=data.get("iou_threshold"),
                    return_annotated=data.get("return_image", False)
                )
                
                await websocket.send_json({
                    "type": "inference_result",
                    "request_id": request_id,
                    "data": response.model_dump(mode="json")
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "request_id": request_id,
                    "error": str(e)
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


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
    logger.info(f"  Cache Enabled: {settings.cache_enabled}")
    logger.info("=" * 60)
    
    logger.info(f"API docs available at: http://{settings.host}:{settings.port}{settings.api_prefix}/docs")
    logger.info(f"n8n webhook endpoint: http://{settings.host}:{settings.port}{settings.n8n_webhook_path}")
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=False
    )


if __name__ == "__main__":
    run_server()