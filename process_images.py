#!/usr/bin/env python3
"""
Batch process images from SAI dataset and generate annotated outputs
"""
import os
import requests
import base64
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# Configuration
SOURCE_DIR = "/mnt/n8n-data/SAINet/predicts/img/"
OUTPUT_DIR = "./out"
API_URL = "http://localhost:8888/api/v1/infer"
MAX_WORKERS = 4  # Concurrent requests
BATCH_SIZE = 10  # Process in batches for progress reporting

def check_api_health():
    """Check if the SAI API is healthy and ready"""
    try:
        health_url = API_URL.replace('/infer', '/health')
        response = requests.get(health_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return True, {
                'status': data.get('status', 'unknown'),
                'model_loaded': data.get('model_loaded', False),
                'model_name': data.get('model_info', {}).get('name', 'unknown'),
                'device': data.get('device', 'unknown'),
                'uptime_seconds': data.get('uptime_seconds', 0)
            }
        else:
            return False, f"Health check failed with status {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return False, "Connection refused - is the SAI service running?"
    except requests.exceptions.Timeout:
        return False, "Health check timeout - service may be overloaded"
    except Exception as e:
        return False, f"Health check error: {e}"

def encode_image(image_path):
    """Encode image to base64"""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding {image_path}: {e}")
        return None

def process_single_image(image_path, output_dir):
    """Process a single image through the API"""
    try:
        # Encode image
        image_b64 = encode_image(image_path)
        if not image_b64:
            return False, f"Failed to encode {image_path}"
        
        # Prepare API request
        payload = {
            'image': image_b64,
            'return_image': True,
            'confidence_threshold': 0.15,  # SAINet2.1 optimized threshold
            'metadata': {'source_file': os.path.basename(image_path)}
        }
        
        # Make API request
        response = requests.post(API_URL, json=payload, timeout=60)
        if response.status_code != 200:
            return False, f"API error {response.status_code} for {image_path}"
        
        data = response.json()
        
        # Create output filename
        input_name = Path(image_path).stem
        output_path = Path(output_dir) / f"{input_name}_annotated.jpg"
        
        # Save annotated image if available
        if data.get('annotated_image'):
            annotated_data = base64.b64decode(data['annotated_image'])
            with open(output_path, 'wb') as f:
                f.write(annotated_data)
        
        # Save detection results as JSON
        results = {
            'source_file': os.path.basename(image_path),
            'processing_time_ms': data.get('processing_time_ms', 0),
            'detection_count': data.get('detection_count', 0),
            'has_fire': data.get('has_fire', False),
            'has_smoke': data.get('has_smoke', False),
            'detections': data.get('detections', []),
            'confidence_scores': data.get('confidence_scores', {}),
            'image_size': data.get('image_size', {}),
            'version': data.get('version', 'unknown')
        }
        
        json_path = Path(output_dir) / f"{input_name}_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Return success with detection info
        detection_info = f"🔥{data.get('detection_count', 0)} detections"
        if data.get('has_fire'):
            detection_info += " [FIRE]"
        if data.get('has_smoke'):
            detection_info += " [SMOKE]"
        
        return True, detection_info
        
    except Exception as e:
        return False, f"Error processing {image_path}: {e}"

def main():
    parser = argparse.ArgumentParser(description='Process SAI images with fire/smoke detection')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS, 
                       help=f'Maximum concurrent workers (default: {MAX_WORKERS})')
    parser.add_argument('--limit', type=int, help='Limit number of images to process (for testing)')
    parser.add_argument('--pattern', type=str, default='*.jpg', help='File pattern to match')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Get list of images to process
    source_path = Path(SOURCE_DIR)
    image_files = list(source_path.glob(args.pattern))
    
    if not image_files:
        print(f"No images found matching pattern '{args.pattern}' in {SOURCE_DIR}")
        return
    
    if args.limit:
        image_files = image_files[:args.limit]
        print(f"Processing limited set: {len(image_files)} images")
    
    print(f"🔥 SAI Image Processing Pipeline")
    print(f"═══════════════════════════════")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images: {len(image_files)} files")
    print(f"Workers: {args.max_workers}")
    print(f"API: {API_URL}")
    print()
    
    # Health check before processing
    print("🏥 Checking API health...")
    healthy, health_info = check_api_health()
    
    if not healthy:
        print(f"❌ API Health Check Failed: {health_info}")
        print("Please ensure the SAI service is running:")
        print("  python run.py")
        print("  or")  
        print("  uvicorn src.main:app --host 0.0.0.0 --port 8888")
        return
    
    print(f"✅ API is healthy!")
    print(f"   Status: {health_info['status']}")
    print(f"   Model: {health_info['model_name']} ({'loaded' if health_info['model_loaded'] else 'not loaded'})")
    print(f"   Device: {health_info['device']}")
    print(f"   Uptime: {health_info['uptime_seconds']}s")
    print()
    
    # Process images
    start_time = time.time()
    processed = 0
    successful = 0
    detections_total = 0
    fires_detected = 0
    smoke_detected = 0
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all jobs
        futures = {
            executor.submit(process_single_image, str(img_path), OUTPUT_DIR): img_path
            for img_path in image_files
        }
        
        # Process completed jobs
        for future in as_completed(futures):
            img_path = futures[future]
            processed += 1
            
            try:
                success, message = future.result()
                if success:
                    successful += 1
                    # Extract detection info from message
                    if "detections" in message:
                        count = int(message.split('🔥')[1].split(' ')[0]) if '🔥' in message else 0
                        detections_total += count
                        if "[FIRE]" in message:
                            fires_detected += 1
                        if "[SMOKE]" in message:
                            smoke_detected += 1
                    
                    print(f"✅ [{processed:3d}/{len(image_files)}] {img_path.name} - {message}")
                else:
                    print(f"❌ [{processed:3d}/{len(image_files)}] {message}")
                    
            except Exception as e:
                print(f"❌ [{processed:3d}/{len(image_files)}] Exception: {e}")
            
            # Progress update every 50 images
            if processed % 50 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(image_files) - processed) / rate if rate > 0 else 0
                print(f"\n📊 Progress: {processed}/{len(image_files)} ({processed/len(image_files)*100:.1f}%)")
                print(f"⚡ Rate: {rate:.1f} images/sec | ETA: {eta/60:.1f} minutes\n")
    
    # Final summary
    elapsed = time.time() - start_time
    print(f"\n🎯 PROCESSING COMPLETE")
    print(f"═══════════════════════")
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {processed - successful}")
    print(f"Processing time: {elapsed:.1f} seconds")
    print(f"Average rate: {len(image_files)/elapsed:.1f} images/sec")
    print()
    print(f"🔥 DETECTION RESULTS")
    print(f"═══════════════════════")
    print(f"Total detections: {detections_total}")
    print(f"Images with fire: {fires_detected}")
    print(f"Images with smoke: {smoke_detected}")
    print(f"Detection rate: {detections_total/successful*100:.1f}% avg detections per image" if successful > 0 else "N/A")
    
    # Generate video from annotated images
    print()
    print(f"🎬 GENERATING VIDEO")
    print(f"══════════════════")
    
    try:
        import subprocess
        import os
        
        # Change to output directory for video generation
        os.chdir(OUTPUT_DIR)
        
        # Create input file list sorted chronologically by timestamp in filename
        # Extract timestamp pattern: sdis-07_brison-200_2024-01-26T15-33-06
        print("📝 Creating chronologically sorted file list...")
        
        # Get all annotated images
        annotated_files = [f for f in os.listdir('.') if f.endswith('_annotated.jpg')]
        
        if annotated_files:
            # Sort by timestamp extracted from filename
            def extract_timestamp(filename):
                # Extract timestamp like "2024-01-26T15-33-06" from filename
                import re
                match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', filename)
                return match.group(1) if match else filename
            
            annotated_files.sort(key=extract_timestamp)
            
            # Create FFmpeg input file
            with open('input.txt', 'w') as f:
                for img_file in annotated_files:
                    f.write(f"file '{img_file}'\n")
            
            # Generate video with FFmpeg
            video_name = f"sai_detection_results_{len(annotated_files)}images.mp4"
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-f', 'concat',
                '-safe', '0',
                '-i', 'input.txt',
                '-c:v', 'libx264',
                '-r', '12',  # 12 FPS for smooth playback
                '-pix_fmt', 'yuv420p',
                '-crf', '18',  # High quality
                video_name
            ]
            
            print(f"🎥 Running FFmpeg to create video: {video_name}")
            print(f"   Frame rate: 12 FPS (smooth playback)")
            print(f"   Images: {len(annotated_files)} chronologically sorted")
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                video_path = os.path.join(OUTPUT_DIR, video_name)
                print(f"✅ Video generated successfully!")
                print(f"📹 Video location: {video_path}")
                print(f"🎬 Duration: ~{len(annotated_files)/12:.1f} seconds at 12 FPS")
                
                # Clean up temp file
                os.remove('input.txt')
            else:
                print(f"❌ FFmpeg error: {result.stderr}")
        else:
            print("⚠️ No annotated images found for video generation")
            
    except ImportError:
        print("⚠️ subprocess module not available")
    except FileNotFoundError:
        print("⚠️ FFmpeg not found - install with: sudo apt-get install ffmpeg")
    except Exception as e:
        print(f"❌ Video generation error: {e}")
    
    print()
    print(f"📁 OUTPUT LOCATION: {OUTPUT_DIR}")
    print(f"   - Annotated images: *_annotated.jpg")
    print(f"   - Detection results: *_results.json")
    print(f"   - Video summary: sai_detection_results_*images.mp4")

if __name__ == "__main__":
    main()