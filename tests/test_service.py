#!/usr/bin/env python3
"""
Test script for SAI Inference Service
"""
import requests
import base64
import json
import time
import argparse
from pathlib import Path
import io
from PIL import Image, ImageDraw
import numpy as np


def get_test_image(category='smoke'):
    """Get a real test image from the tests/images directory"""
    test_dir = Path(__file__).parent / 'images' / category
    
    # Find first available image
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images = list(test_dir.glob(ext))
        if images:
            image_path = images[0]
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            return {
                'bytes': image_bytes,
                'base64': base64.b64encode(image_bytes).decode('utf-8'),
                'filename': image_path.name
            }
    
    # Fallback to synthetic image if no real images found
    return create_synthetic_image()


def create_synthetic_image(width=896, height=896):
    """Create a synthetic test image as fallback"""
    # Create RGB image
    img = Image.new('RGB', (width, height), color='blue')
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes to simulate fire/smoke
    draw.rectangle([100, 100, 300, 300], fill='red', outline='orange')  # Fire
    draw.ellipse([500, 200, 700, 400], fill='gray', outline='white')   # Smoke
    
    # Convert to binary and base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    image_bytes = buffer.getvalue()
    
    return {
        'bytes': image_bytes,
        'base64': base64.b64encode(image_bytes).decode('utf-8'),
        'filename': 'synthetic_test.jpg'
    }


def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8888/api/v1/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Model loaded: {data['is_model_loaded']}")
            print(f"   Version: {data['version']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_models():
    """Test models endpoint"""
    print("\\nTesting models endpoint...")
    try:
        response = requests.get("http://localhost:8888/api/v1/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Models endpoint working")
            print(f"   Current model: {data.get('current_model', 'None')}")
            print(f"   Loaded models: {len(data.get('loaded_models', []))}")
            return True
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint failed: {e}")
        return False


def test_inference(use_base64=False):
    """Test inference endpoint - multipart/form-data (default) or base64 (with flag)"""
    endpoint = "/api/v1/infer/base64" if use_base64 else "/api/v1/infer"
    method = "base64" if use_base64 else "multipart/form-data"
    
    print(f"\\nTesting inference endpoint ({method})...")
    try:
        # Get test image
        test_image = get_test_image('smoke')
        
        start_time = time.time()
        
        if use_base64:
            # Test base64 endpoint with JSON
            payload = {
                "image": test_image['base64'],
                "confidence_threshold": 0.25,
                "return_image": False
            }
            
            response = requests.post(
                f"http://localhost:8888{endpoint}",
                json=payload,
                timeout=30
            )
        else:
            # Test primary endpoint with multipart/form-data  
            files = {
                'file': (test_image['filename'], test_image['bytes'], 'image/jpeg')
            }
            data = {
                'confidence_threshold': 0.25,
                'return_image': 'false'
            }
            
            response = requests.post(
                f"http://localhost:8888{endpoint}",
                files=files,
                data=data,
                timeout=30
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Inference successful ({method})")
            print(f"   Image: {test_image['filename']}")
            print(f"   Processing time: {processing_time:.1f}ms (client)")
            print(f"   Server time: {data.get('processing_time_ms', 0):.1f}ms")
            print(f"   Detections: {len(data.get('detections', []))}")
            print(f"   Has fire: {data.get('has_fire', False)}")
            print(f"   Has smoke: {data.get('has_smoke', False)}")
            
            if data.get('detections'):
                for i, det in enumerate(data['detections']):
                    print(f"   Detection {i+1}: {det['class_name']} ({det['confidence']:.3f})")
            
            return True
        else:
            print(f"❌ Inference failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False


def test_webhook():
    """Test webhook functionality via base64 endpoint"""
    print("\\nTesting webhook functionality...")
    try:
        test_image = get_test_image('fire')  # Use a fire image for more likely detection
        
        payload = {
            "image": test_image['base64'],
            "confidence_threshold": 0.25,
            "webhook_url": "https://httpbin.org/post",  # Test webhook URL
            "metadata": {
                "workflow_id": "test_workflow",
                "execution_id": "test_execution"
            }
        }
        
        response = requests.post(
            "http://localhost:8888/api/v1/infer/base64",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Webhook test successful")
            print(f"   Image: {test_image['filename']}")
            print(f"   Detections: {len(data.get('detections', []))}")
            print(f"   Has fire: {data.get('has_fire', False)}")
            print(f"   Has smoke: {data.get('has_smoke', False)}")
            print(f"   Alert level: {data.get('alert_level', 'none')}")
            return True
        else:
            print(f"❌ Webhook test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Webhook test failed: {e}")
        return False


def test_smoke_only_detection():
    """Test smoke-only detection using detection_classes parameter"""
    print("\\nTesting smoke-only detection (detection_classes=[0])...")
    try:
        # Use a mixed fire/smoke image to verify filtering works
        test_image = get_test_image('both') if Path(__file__).parent.joinpath('images/both').exists() else get_test_image('fire')
        
        # Test 1: Regular detection (both classes)
        print("   Step 1: Testing normal detection (all classes)...")
        normal_payload = {
            "image": test_image['base64'],
            "confidence_threshold": 0.25,
            "return_image": False
        }
        
        normal_response = requests.post(
            "http://localhost:8888/api/v1/infer/base64",
            json=normal_payload,
            timeout=30
        )
        
        if normal_response.status_code != 200:
            print(f"❌ Normal detection failed: {normal_response.status_code}")
            return False
            
        normal_data = normal_response.json()
        normal_detections = normal_data.get('detections', [])
        
        # Test 2: Smoke-only detection
        print("   Step 2: Testing smoke-only detection (detection_classes=[0])...")
        smoke_payload = {
            "image": test_image['base64'],
            "confidence_threshold": 0.25,
            "detection_classes": [0],  # 0 = smoke only
            "return_image": False
        }
        
        smoke_response = requests.post(
            "http://localhost:8888/api/v1/infer/base64",
            json=smoke_payload,
            timeout=30
        )
        
        if smoke_response.status_code != 200:
            print(f"❌ Smoke-only detection failed: {smoke_response.status_code}")
            print(f"   Response: {smoke_response.text}")
            return False
        
        smoke_data = smoke_response.json()
        smoke_detections = smoke_data.get('detections', [])
        
        # Validate results
        print(f"✅ Smoke-only detection successful")
        print(f"   Image: {test_image['filename']}")
        print(f"   Normal detection: {len(normal_detections)} total detections")
        
        # Count by class in normal detection
        normal_smoke_count = sum(1 for d in normal_detections if d.get('class_id') == 0)
        normal_fire_count = sum(1 for d in normal_detections if d.get('class_id') == 1)
        print(f"   ├─ Smoke: {normal_smoke_count}, Fire: {normal_fire_count}")
        
        print(f"   Smoke-only detection: {len(smoke_detections)} detections")
        
        # Validate all detections are smoke (class_id = 0)
        smoke_only_count = sum(1 for d in smoke_detections if d.get('class_id') == 0)
        fire_in_smoke = sum(1 for d in smoke_detections if d.get('class_id') == 1)
        
        print(f"   ├─ Smoke: {smoke_only_count}, Fire: {fire_in_smoke}")
        
        # Validation checks
        if fire_in_smoke > 0:
            print(f"❌ FAIL: Found {fire_in_smoke} fire detections when filtering for smoke only!")
            return False
            
        if smoke_only_count != len(smoke_detections):
            print(f"❌ FAIL: Some detections have unexpected class IDs")
            return False
            
        if len(smoke_detections) > normal_smoke_count:
            print(f"❌ FAIL: Smoke-only returned more smoke than normal detection")
            return False
            
        # Show individual smoke detections
        if smoke_detections:
            print("   └─ Smoke detections:")
            for i, det in enumerate(smoke_detections):
                print(f"      {i+1}. {det['class_name']} (confidence: {det['confidence']:.3f})")
        
        print(f"✅ Validation passed: All {len(smoke_detections)} detections are smoke-only")
        return True
        
    except Exception as e:
        print(f"❌ Smoke-only detection failed: {e}")
        return False


def test_batch_inference():
    """Test batch inference"""
    print("\\nTesting batch inference...")
    try:
        # Create multiple test images from different categories
        images = [
            get_test_image('smoke')['base64'],
            get_test_image('fire')['base64'],
            get_test_image('both')['base64'] if Path(__file__).parent.joinpath('images/both').exists() else get_test_image('smoke')['base64']
        ]
        
        payload = {
            "images": images,
            "confidence_threshold": 0.25,
            "parallel_processing": True
        }
        
        start_time = time.time()
        response = requests.post(
            "http://localhost:8888/api/v1/infer/batch",
            json=payload,
            timeout=60
        )
        processing_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Batch inference successful")
            print(f"   Processing time: {processing_time:.1f}ms (client)")
            print(f"   Server time: {data.get('total_processing_time_ms', 0):.1f}ms")
            print(f"   Total detections: {data.get('total_detections', 0)}")
            print(f"   Images processed: {len(data.get('results', []))}")
            
            # Show individual results
            for i, result in enumerate(data.get('results', [])):
                print(f"   Image {i+1}: {len(result.get('detections', []))} detections")
            
            return True
        else:
            print(f"❌ Batch inference failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch inference failed: {e}")
        return False


def main():
    """Run all tests"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SAI Inference Service Test Suite')
    parser.add_argument('--base64', action='store_true', 
                       help='Test base64 endpoint instead of multipart/form-data')
    parser.add_argument('--both', action='store_true',
                       help='Test both multipart and base64 endpoints')
    parser.add_argument('--smoke-only', action='store_true',
                       help='Run only the smoke-only detection test')
    args = parser.parse_args()
    
    print("🔥 SAI Inference Service Test Suite")
    print("===================================")
    
    # Handle smoke-only test flag
    if args.smoke_only:
        all_tests = [("Smoke-Only Detection", lambda: test_smoke_only_detection())]
    else:
        # Define base tests
        base_tests = [
            ("Health Check", lambda: test_health()),
            ("Models API", lambda: test_models()),
        ]
        
        # Define inference tests based on flags
        inference_tests = []
        
        if args.both:
            inference_tests.extend([
                ("Single Inference (multipart)", lambda: test_inference(False)),
                ("Single Inference (base64)", lambda: test_inference(True)),
            ])
        elif args.base64:
            inference_tests.append(("Single Inference (base64)", lambda: test_inference(True)))
        else:
            inference_tests.append(("Single Inference (multipart)", lambda: test_inference(False)))
        
        # Add remaining tests
        other_tests = [
            ("Webhook Functionality", lambda: test_webhook()),
            ("Smoke-Only Detection", lambda: test_smoke_only_detection()),
            ("Batch Inference", lambda: test_batch_inference()),
        ]
        
        # Combine all tests
        all_tests = base_tests + inference_tests + other_tests
    
    passed = 0
    total = len(all_tests)
    
    # Run tests
    for test_name, test_func in all_tests:
        print(f"\\n{test_name}:")
        print("-" * len(test_name))
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
        time.sleep(0.5)  # Brief pause between tests
    
    print(f"\\n{'='*50}")
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Service is working correctly.")
        return 0
    elif passed > 0:
        print("⚠️  Some tests failed. Check the service configuration.")
        print("💡 Try running with different flags:")
        print("   --base64: Test base64 endpoint")
        print("   --both: Test both endpoints")
        return 1
    else:
        print("❌ All tests failed. Check if the service is running.")
        return 2


if __name__ == "__main__":
    exit(main())