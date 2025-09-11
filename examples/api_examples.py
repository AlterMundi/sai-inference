#!/usr/bin/env python3
"""
SAI Inference Service - API Usage Examples
Demonstrates all endpoint usage patterns with annotated images
"""

import requests
import base64
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw
import io


# Service configuration
API_BASE = "http://localhost:8888"
API_V1 = f"{API_BASE}/api/v1"


def create_test_image(width=1920, height=1080, with_fire=True):
    """Create a test image with simulated fire/smoke"""
    img = Image.new('RGB', (width, height), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    if with_fire:
        # Draw fire (red/orange rectangle)
        draw.rectangle([400, 300, 600, 500], fill='red', outline='orange', width=3)
        draw.rectangle([420, 320, 580, 480], fill='orange')
        
        # Draw smoke (gray ellipse)
        draw.ellipse([300, 200, 500, 350], fill='gray', outline='darkgray', width=2)
        
        # Add some flames
        for i in range(5):
            x = 450 + i * 20
            draw.polygon([(x, 500), (x+10, 480), (x+20, 500), (x+15, 520), (x+5, 520)], 
                        fill='yellow', outline='red')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def test_health_check():
    """Test service health and model status"""
    print("🔍 Testing Health Check...")
    
    try:
        response = requests.get(f"{API_V1}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Service healthy")
            print(f"   Status: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   Version: {data['version']}")
            
            if data.get('model_info'):
                model = data['model_info']
                print(f"   Model: {model.get('name', 'unknown')}")
                print(f"   Architecture: {model.get('architecture', 'unknown')}")
                print(f"   Input size: {model.get('input_size', 'unknown')}px")
            
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def example_basic_inference():
    """Basic inference example without annotation"""
    print("\\n🔥 Example 1: Basic Inference (no annotation)")
    print("-" * 50)
    
    # Create test image
    image_b64 = create_test_image(with_fire=True)
    
    payload = {
        "image": image_b64,
        "confidence_threshold": 0.15,  # SAINet2.1 reference value
        "return_image": False,  # No annotation for speed
        "metadata": {
            "example": "basic_inference",
            "camera": "test_camera"
        }
    }
    
    start_time = time.time()
    response = requests.post(f"{API_V1}/infer", json=payload, timeout=30)
    request_time = (time.time() - start_time) * 1000
    
    if response.status_code == 200:
        result = response.json()
        
        print("✅ Inference successful")
        print(f"   Request time: {request_time:.1f}ms")
        print(f"   Processing time: {result['processing_time_ms']:.1f}ms") 
        print(f"   Image size: {result['image_size']['width']}x{result['image_size']['height']}")
        print(f"   Detections: {result['detection_count']}")
        print(f"   Has fire: {result['has_fire']}")
        print(f"   Has smoke: {result['has_smoke']}")
        
        # Show detection details
        for i, det in enumerate(result['detections']):
            bbox = det['bbox']
            print(f"   Detection {i+1}: {det['class_name']} ({det['confidence']:.3f})")
            print(f"     Location: ({bbox['x1']:.1f}, {bbox['y1']:.1f}) -> ({bbox['x2']:.1f}, {bbox['y2']:.1f})")
            print(f"     Size: {bbox['width']:.1f} x {bbox['height']:.1f}px")
        
        return result
    else:
        print(f"❌ Inference failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None


def example_annotated_inference():
    """Inference with annotated image output"""
    print("\\n🎨 Example 2: Inference with Annotated Image")
    print("-" * 50)
    
    # Create test image
    image_b64 = create_test_image(width=1280, height=720, with_fire=True)
    
    payload = {
        "image": image_b64,
        "confidence_threshold": 0.15,
        "return_image": True,  # ⭐ Request annotated image
        "metadata": {
            "example": "annotated_inference",
            "request_annotations": True
        }
    }
    
    start_time = time.time()
    response = requests.post(f"{API_V1}/infer", json=payload, timeout=60)
    request_time = (time.time() - start_time) * 1000
    
    if response.status_code == 200:
        result = response.json()
        
        print("✅ Annotated inference successful")
        print(f"   Request time: {request_time:.1f}ms")
        print(f"   Processing time: {result['processing_time_ms']:.1f}ms")
        print(f"   Detections: {result['detection_count']}")
        
        # Check for annotated image
        if result.get('annotated_image'):
            print("   📸 Annotated image received!")
            
            # Decode and save annotated image
            try:
                # Handle data URI format
                if result['annotated_image'].startswith('data:'):
                    image_data = result['annotated_image'].split(',')[1]
                else:
                    image_data = result['annotated_image']
                
                image_bytes = base64.b64decode(image_data)
                
                # Save to file
                output_path = Path("examples") / "annotated_output.jpg"
                output_path.parent.mkdir(exist_ok=True)
                
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                
                print(f"   💾 Saved annotated image: {output_path}")
                print(f"   📏 Annotated image size: {len(image_bytes) / 1024:.1f} KB")
                
                # Verify image can be loaded
                test_img = Image.open(io.BytesIO(image_bytes))
                print(f"   ✅ Image verified: {test_img.size[0]}x{test_img.size[1]}")
                
            except Exception as e:
                print(f"   ❌ Error saving annotated image: {e}")
        else:
            print("   ⚠️ No annotated image in response")
        
        return result
    else:
        print(f"❌ Annotated inference failed: {response.status_code}")
        return None


def example_file_upload():
    """File upload inference example"""
    print("\\n📁 Example 3: File Upload Inference")
    print("-" * 50)
    
    # Create and save test image
    image_b64 = create_test_image(with_fire=True)
    image_bytes = base64.b64decode(image_b64)
    
    test_file = Path("examples") / "test_fire_scene.jpg"
    test_file.parent.mkdir(exist_ok=True)
    
    with open(test_file, "wb") as f:
        f.write(image_bytes)
    
    print(f"   📄 Created test file: {test_file}")
    
    # Upload file
    try:
        with open(test_file, "rb") as f:
            files = {"file": ("fire_scene.jpg", f, "image/jpeg")}
            data = {
                "confidence_threshold": 0.15,
                "return_image": "true"  # String for form data
            }
            
            start_time = time.time()
            response = requests.post(f"{API_V1}/infer/file", files=files, data=data, timeout=60)
            request_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ File upload inference successful")
            print(f"   Request time: {request_time:.1f}ms")
            print(f"   Processing time: {result['processing_time_ms']:.1f}ms")
            print(f"   File processed: {result['metadata'].get('filename', 'unknown')}")
            print(f"   Detections: {result['detection_count']}")
            
            # Save annotated result if present
            if result.get('annotated_image'):
                image_data = result['annotated_image'].split(',')[1] if result['annotated_image'].startswith('data:') else result['annotated_image']
                
                output_path = Path("examples") / "file_upload_result.jpg"
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(image_data))
                
                print(f"   💾 Saved result: {output_path}")
            
            return result
        else:
            print(f"❌ File upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ File upload error: {e}")
        return None


def example_batch_inference():
    """Batch processing example"""
    print("\\n📦 Example 4: Batch Inference")
    print("-" * 50)
    
    # Create multiple test images
    images = [
        create_test_image(with_fire=True),   # Fire scene
        create_test_image(with_fire=False),  # Normal scene
        create_test_image(width=800, height=600, with_fire=True)  # Another fire scene
    ]
    
    payload = {
        "images": images,
        "confidence_threshold": 0.15,
        "return_images": True,  # ⭐ Return all annotated images
        "parallel_processing": True,
        "metadata": {
            "example": "batch_processing",
            "batch_size": len(images)
        }
    }
    
    start_time = time.time()
    response = requests.post(f"{API_V1}/infer/batch", json=payload, timeout=120)
    request_time = (time.time() - start_time) * 1000
    
    if response.status_code == 200:
        result = response.json()
        
        print("✅ Batch inference successful")
        print(f"   Request time: {request_time:.1f}ms")
        print(f"   Total processing time: {result['total_processing_time_ms']:.1f}ms")
        print(f"   Images processed: {len(result['results'])}")
        print(f"   Total detections: {result['total_detections']}")
        
        # Process each result
        for i, img_result in enumerate(result['results']):
            print(f"   \\n   Image {i+1}:")
            print(f"     Detections: {img_result['detection_count']}")
            print(f"     Has fire: {img_result['has_fire']}")
            print(f"     Has smoke: {img_result['has_smoke']}")
            print(f"     Processing time: {img_result['processing_time_ms']:.1f}ms")
            
            # Save annotated image
            if img_result.get('annotated_image'):
                image_data = img_result['annotated_image'].split(',')[1] if img_result['annotated_image'].startswith('data:') else img_result['annotated_image']
                
                output_path = Path("examples") / f"batch_result_{i+1}.jpg"
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(image_data))
                
                print(f"     💾 Saved: {output_path}")
        
        # Show summary
        summary = result['summary']
        print(f"   \\n   📊 Summary:")
        print(f"     Total images: {summary['total_images']}")
        print(f"     Images with fire: {summary['images_with_fire']}")
        print(f"     Images with smoke: {summary['images_with_smoke']}")
        print(f"     Average detections: {summary['average_detections_per_image']:.2f}")
        
        return result
    else:
        print(f"❌ Batch inference failed: {response.status_code}")
        return None


def example_n8n_webhook():
    """n8n webhook integration example"""
    print("\\n🔗 Example 5: n8n Webhook Integration")
    print("-" * 50)
    
    # Create fire scene for alert
    image_b64 = create_test_image(with_fire=True)
    
    payload = {
        "image": image_b64,
        "confidence_threshold": 0.15,
        "return_image": True,
        "workflow_id": "fire_detection_demo",
        "execution_id": f"exec_{int(time.time())}",
        "metadata": {
            "camera": "warehouse_camera_01",
            "location": "Building A - Section 3",
            "priority": "critical"
        }
    }
    
    start_time = time.time()
    response = requests.post(f"{API_BASE}/api/v1/infer/base64", json=payload, timeout=60)
    request_time = (time.time() - start_time) * 1000
    
    if response.status_code == 200:
        result = response.json()
        
        print("✅ n8n webhook successful")
        print(f"   Request time: {request_time:.1f}ms")
        print(f"   Success: {result['success']}")
        print(f"   Request ID: {result['request_id']}")
        print(f"   Detections: {result['detections']}")
        print(f"   Has fire: {result['has_fire']}")
        print(f"   Alert level: {result['alert_level']}")
        
        # Check full data
        if result.get('data'):
            data = result['data']
            print(f"   Processing time: {data['processing_time_ms']:.1f}ms")
            
            # Save webhook annotated image
            if data.get('annotated_image'):
                image_data = data['annotated_image'].split(',')[1] if data['annotated_image'].startswith('data:') else data['annotated_image']
                
                output_path = Path("examples") / "webhook_alert.jpg"
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(image_data))
                
                print(f"   💾 Alert image saved: {output_path}")
        
        return result
    else:
        print(f"❌ n8n webhook failed: {response.status_code}")
        return None


def main():
    """Run all API examples"""
    print("🔥 SAI Inference Service - API Examples")
    print("=" * 60)
    
    # Ensure examples directory exists
    Path("examples").mkdir(exist_ok=True)
    
    examples = [
        ("Health Check", test_health_check),
        ("Basic Inference", example_basic_inference),
        ("Annotated Inference", example_annotated_inference),
        ("File Upload", example_file_upload),
        ("Batch Processing", example_batch_inference),
        ("n8n Webhook", example_n8n_webhook)
    ]
    
    results = {}
    
    for name, example_func in examples:
        print(f"\\n{'='*60}")
        try:
            result = example_func()
            results[name] = result
            print(f"✅ {name} completed")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = None
        
        time.sleep(1)  # Brief pause between examples
    
    # Summary
    print(f"\\n{'='*60}")
    print("📊 EXAMPLES SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results.values() if r is not None)
    total = len(results)
    
    print(f"Completed: {successful}/{total} examples")
    
    for name, result in results.items():
        status = "✅" if result is not None else "❌"
        print(f"  {status} {name}")
    
    if successful == total:
        print("\\n🎉 All examples completed successfully!")
        print("\\n📁 Generated files in examples/ directory:")
        examples_dir = Path("examples")
        if examples_dir.exists():
            for file in examples_dir.glob("*"):
                print(f"  📄 {file.name}")
    else:
        print(f"\\n⚠️  {total - successful} examples failed. Check service status.")
    
    return 0 if successful == total else 1


if __name__ == "__main__":
    exit(main())