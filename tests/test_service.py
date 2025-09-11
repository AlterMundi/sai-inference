#!/usr/bin/env python3
"""
Test script for SAI Inference Service
"""
import requests
import base64
import json
import time
from pathlib import Path
import io
from PIL import Image, ImageDraw
import numpy as np


def create_test_image(width=896, height=896):
    """Create a simple test image"""
    # Create RGB image
    img = Image.new('RGB', (width, height), color='blue')
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes to simulate fire/smoke
    draw.rectangle([100, 100, 300, 300], fill='red', outline='orange')  # Fire
    draw.ellipse([500, 200, 700, 400], fill='gray', outline='white')   # Smoke
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode('utf-8')


def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8888/api/v1/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
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


def test_inference():
    """Test inference endpoint"""
    print("\\nTesting inference endpoint...")
    try:
        # Create test image
        test_image = create_test_image()
        
        payload = {
            "image": test_image,
            "confidence_threshold": 0.25,
            "return_image": False
        }
        
        start_time = time.time()
        response = requests.post(
            "http://localhost:8888/api/v1/infer",
            json=payload,
            timeout=30
        )
        processing_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Inference successful")
            print(f"   Processing time: {processing_time:.1f}ms (client)")
            print(f"   Server time: {data.get('processing_time_ms', 0):.1f}ms")
            print(f"   Detections: {data.get('detection_count', 0)}")
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
    """Test n8n webhook endpoint"""
    print("\\nTesting n8n webhook endpoint...")
    try:
        test_image = create_test_image()
        
        payload = {
            "image": test_image,
            "confidence_threshold": 0.25,
            "workflow_id": "test_workflow",
            "execution_id": "test_execution"
        }
        
        response = requests.post(
            "http://localhost:8888/api/v1/infer/base64",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Webhook successful")
            print(f"   Success: {data.get('success', False)}")
            print(f"   Detections: {data.get('detections', 0)}")
            print(f"   Alert level: {data.get('alert_level', 'none')}")
            return True
        else:
            print(f"❌ Webhook failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Webhook failed: {e}")
        return False


def test_batch_inference():
    """Test batch inference"""
    print("\\nTesting batch inference...")
    try:
        # Create multiple test images
        images = [create_test_image() for _ in range(3)]
        
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
            return True
        else:
            print(f"❌ Batch inference failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Batch inference failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🔥 SAI Inference Service Test Suite")
    print("===================================")
    
    tests = [
        ("Health Check", test_health),
        ("Models API", test_models),
        ("Single Inference", test_inference),
        ("n8n Webhook", test_webhook),
        ("Batch Inference", test_batch_inference),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\\n{test_name}:")
        print("-" * len(test_name))
        if test_func():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print(f"\\n{'='*40}")
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Service is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the service configuration.")
        return 1


if __name__ == "__main__":
    exit(main())