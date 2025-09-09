#!/usr/bin/env python3
"""
SAI Daily Test Service
Sends test images to n8n webhook for end-to-end alert system validation
"""
import os
import sys
import json
import logging
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class DailyTestConfig:
    """Configuration for daily test service"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Store the config file path for later reference
        self.config_file_path = None
        
        # Load configuration from environment file
        if config_file and Path(config_file).exists():
            load_dotenv(config_file, override=True)
            self.config_file_path = config_file
        else:
            # Try default locations
            default_configs = [
                "/etc/sai-inference/daily-test.env",
                "./config/daily-test.env",
                "./.env"
            ]
            for config_path in default_configs:
                if Path(config_path).exists():
                    load_dotenv(config_path, override=True)
                    self.config_file_path = config_path
                    break
                    
        # Set fallback if no config file was found
        if self.config_file_path is None:
            self.config_file_path = "/etc/sai-inference/daily-test.env"
        
        # n8n Configuration
        self.webhook_url = os.getenv("N8N_WEBHOOK_URL", "")
        self.auth_token = os.getenv("N8N_AUTH_TOKEN", "")
        
        # Test Images Configuration - resolve relative to installation root
        install_root = Path(__file__).parent.parent  # /opt/sai-inference
        test_images_path = os.getenv("TEST_IMAGES_DIR", "tests/images")
        test_labels_path = os.getenv("TEST_LABELS_DIR", "tests/labels")
        
        # Make paths absolute if they're relative
        if not os.path.isabs(test_images_path):
            self.test_images_dir = install_root / test_images_path
        else:
            self.test_images_dir = Path(test_images_path)
            
        if not os.path.isabs(test_labels_path):
            self.test_labels_dir = install_root / test_labels_path
        else:
            self.test_labels_dir = Path(test_labels_path)
        self.fire_subdir = os.getenv("FIRE_SUBDIR", "fire")
        self.smoke_subdir = os.getenv("SMOKE_SUBDIR", "smoke")
        self.both_subdir = os.getenv("BOTH_SUBDIR", "both")
        
        # Test Metadata
        self.test_location = os.getenv("TEST_LOCATION", "TestLab")
        self.test_device_id = os.getenv("TEST_DEVICE_ID", "sai-test-runner-01")
        self.randomize_metadata = os.getenv("RANDOMIZE_METADATA", "true").lower() == "true"
        
        # Test Categories
        self.enable_fire_tests = os.getenv("ENABLE_FIRE_TESTS", "true").lower() == "true"
        self.enable_smoke_tests = os.getenv("ENABLE_SMOKE_TESTS", "true").lower() == "true"
        self.enable_both_tests = os.getenv("ENABLE_BOTH_TESTS", "true").lower() == "true"
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_file = os.getenv("LOG_FILE", "/var/log/sai-inference/daily-test.log")
        
        # Request Configuration
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        
        # Cron Schedule Configuration
        self.cron_schedule = os.getenv("CRON_SCHEDULE", "0 9,17 * * *")
        self.cron_description = os.getenv("CRON_DESCRIPTION", "Twice daily at 9:00 AM and 5:00 PM")
        self.cron_enabled = os.getenv("CRON_ENABLED", "true").lower() == "true"
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        if not self.webhook_url:
            errors.append("N8N_WEBHOOK_URL not configured")
        
        if not self.auth_token:
            errors.append("N8N_AUTH_TOKEN not configured")
            
        if not self.test_images_dir.exists():
            errors.append(f"Test images directory not found: {self.test_images_dir}")
        
        if not self.test_labels_dir.exists():
            errors.append(f"Test labels directory not found: {self.test_labels_dir}")
        
        return errors


class DailyTestService:
    """Main service for running daily tests"""
    
    def __init__(self, config: DailyTestConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Test locations for randomization
        self.test_locations = [
            "TestLab", "StagingEnv", "ValidationSite", 
            "QA_Station", "TestBench_Alpha", "DevTest_Beta"
        ]
        
        # Device IDs for randomization
        self.test_device_ids = [
            "sai-test-runner-01", "sai-test-runner-02", "sai-test-runner-03",
            "sai-validation-cam-01", "sai-qa-device-01"
        ]
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("sai_daily_test")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Create log directory if it doesn't exist
        log_dir = Path(self.config.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(self.config.log_file)
        file_handler.setLevel(getattr(logging, self.config.log_level))
        
        # Console handler for cron output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def discover_test_images(self) -> Dict[str, List[Path]]:
        """Discover test images in configured directories"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        categories = {}
        
        # Check each category directory
        for category, subdir in [
            ('fire', self.config.fire_subdir),
            ('smoke', self.config.smoke_subdir),
            ('both', self.config.both_subdir)
        ]:
            category_dir = self.config.test_images_dir / subdir
            images = []
            
            if category_dir.exists():
                for image_file in category_dir.iterdir():
                    if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                        images.append(image_file)
            
            categories[category] = images
            self.logger.debug(f"Found {len(images)} images in {category} category")
        
        return categories
    
    def generate_test_metadata(self) -> Dict[str, Any]:
        """Generate metadata for test request"""
        if self.config.randomize_metadata:
            location = random.choice(self.test_locations)
            device_id = random.choice(self.test_device_ids)
        else:
            location = self.config.test_location
            device_id = self.config.test_device_id
        
        return {
            "timestamp": datetime.now().isoformat(),
            "location": location,
            "device_id": device_id
        }
    
    def load_label_metadata(self, image_path: Path, category: str) -> Dict[str, Any]:
        """Load YOLO annotations from corresponding label file if it exists"""
        # Construct label file path with same name but .txt extension
        label_filename = image_path.stem + '.txt'
        label_path = self.config.test_labels_dir / category / label_filename
        
        if label_path.exists():
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    label_content = f.read().strip()
                
                if label_content:
                    # Parse YOLO format annotations
                    annotations = []
                    detection_classes = set()
                    
                    for line in label_content.split('\n'):
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                confidence = float(parts[5]) if len(parts) > 5 else None
                                
                                # Map class IDs to detection types (standard SAI mapping)
                                class_name = "fire" if class_id == 0 else "smoke" if class_id == 1 else f"class_{class_id}"
                                detection_classes.add(class_name)
                                
                                annotation = {
                                    "class_id": class_id,
                                    "class_name": class_name,
                                    "x_center": x_center,
                                    "y_center": y_center,
                                    "width": width,
                                    "height": height
                                }
                                
                                if confidence is not None:
                                    annotation["confidence"] = confidence
                                    
                                annotations.append(annotation)
                    
                    # Create metadata with YOLO annotations
                    label_metadata = {
                        "yolo_annotations": annotations,
                        "annotation_count": len(annotations),
                        "detected_classes": sorted(list(detection_classes)),
                        "has_fire": "fire" in detection_classes,
                        "has_smoke": "smoke" in detection_classes,
                        "label_source": str(label_path)
                    }
                    
                    self.logger.debug(f"Loaded YOLO annotations from {label_path}: {len(annotations)} detections")
                    return label_metadata
                else:
                    self.logger.debug(f"Label file {label_path} is empty")
                    return {}
                    
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Invalid YOLO format in label file {label_path}: {e}")
                return {}
            except Exception as e:
                self.logger.warning(f"Error reading label file {label_path}: {e}")
                return {}
        else:
            self.logger.debug(f"No label file found for {image_path.name} in category {category}")
            return {}
        
        return {}
    
    def send_test_image(self, image_path: Path, category: str) -> bool:
        """Send a single test image to n8n webhook"""
        try:
            # Generate base metadata
            metadata = self.generate_test_metadata()
            
            # Load and merge label metadata if available
            label_metadata = self.load_label_metadata(image_path, category)
            if label_metadata:
                # Merge label metadata into base metadata
                # Label metadata takes precedence for overlapping keys
                metadata.update(label_metadata)
                self.logger.info(f"Merged label metadata for {image_path.name}")
            
            with open(image_path, 'rb') as img_file:
                files = {
                    'image': (image_path.name, img_file, 'image/jpeg'),
                    'metadata': (None, json.dumps(metadata), 'application/json')
                }
                
                headers = {
                    'Authorization': f'Bearer {self.config.auth_token}'
                }
                
                self.logger.info(f"Sending {category} test image: {image_path.name}")
                self.logger.debug(f"Metadata: {metadata}")
                
                response = requests.post(
                    self.config.webhook_url,
                    files=files,
                    headers=headers,
                    timeout=self.config.request_timeout
                )
                
                if response.status_code == 200:
                    self.logger.info(f"✅ {category} test successful - {image_path.name}")
                    self.logger.debug(f"Response: {response.text}")
                    return True
                else:
                    self.logger.error(f"❌ {category} test failed - {image_path.name}: HTTP {response.status_code}")
                    self.logger.error(f"Response: {response.text}")
                    return False
                    
        except requests.exceptions.Timeout:
            self.logger.error(f"❌ {category} test timeout - {image_path.name}")
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ {category} test request failed - {image_path.name}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ {category} test error - {image_path.name}: {e}")
            return False
    
    def run_tests(self) -> Dict[str, Any]:
        """Run all configured tests"""
        self.logger.info("🔥 Starting SAI Daily Test Service")
        self.logger.info(f"Webhook URL: {self.config.webhook_url}")
        
        # Check cron configuration on every run
        self._check_cron_configuration()
        
        # Discover test images
        test_images = self.discover_test_images()
        
        results = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'categories_tested': [],
            'start_time': datetime.now().isoformat()
        }
        
        # Run tests for each enabled category
        for category, enabled in [
            ('fire', self.config.enable_fire_tests),
            ('smoke', self.config.enable_smoke_tests),
            ('both', self.config.enable_both_tests)
        ]:
            if not enabled:
                self.logger.info(f"Skipping {category} tests (disabled)")
                continue
            
            images = test_images.get(category, [])
            if not images:
                self.logger.warning(f"No {category} test images found")
                continue
            
            # Select random image from category
            selected_image = random.choice(images)
            results['total_tests'] += 1
            results['categories_tested'].append(category)
            
            # Send test with retry logic
            success = False
            for attempt in range(1, self.config.max_retries + 1):
                if attempt > 1:
                    self.logger.info(f"Retry attempt {attempt}/{self.config.max_retries}")
                
                success = self.send_test_image(selected_image, category)
                if success:
                    break
            
            if success:
                results['successful_tests'] += 1
            else:
                results['failed_tests'] += 1
        
        results['end_time'] = datetime.now().isoformat()
        
        # Log summary
        self.logger.info("📊 Test Summary:")
        self.logger.info(f"   Total tests: {results['total_tests']}")
        self.logger.info(f"   Successful: {results['successful_tests']}")
        self.logger.info(f"   Failed: {results['failed_tests']}")
        self.logger.info(f"   Categories: {', '.join(results['categories_tested'])}")
        
        if results['failed_tests'] > 0:
            self.logger.error("⚠️ Some tests failed - check alert system health")
        else:
            self.logger.info("🎉 All tests passed - alert system operational")
        
        return results
    
    def _check_cron_configuration(self) -> None:
        """Check if cron configuration matches current settings and warn if not"""
        try:
            status = self.show_cron_status()
            
            if status['needs_update']:
                self.logger.warning("⚠️ Cron configuration mismatch detected!")
                self.logger.warning(f"   Config Schedule: {status['config_schedule']} ({status['config_description']})")
                if status['current_cron_entry']:
                    current_schedule = status['current_cron_entry'].split()[0:5]  # Extract cron fields
                    self.logger.warning(f"   Actual Schedule: {' '.join(current_schedule)}")
                else:
                    self.logger.warning("   Actual Schedule: Not scheduled")
                self.logger.warning(f"   Config Enabled: {status['config_enabled']}")
                self.logger.warning("   Run 'python daily_test.py --update-cron' to sync configuration")
            elif status['config_enabled'] and status['is_scheduled']:
                self.logger.info(f"✓ Cron schedule synchronized: {status['config_description']}")
            elif not status['config_enabled']:
                self.logger.info("✓ Cron scheduling disabled in configuration")
            
        except Exception as e:
            self.logger.warning(f"Could not check cron configuration: {e}")
    
    def get_current_cron_entry(self) -> Optional[str]:
        """Get the current cron entry for this service"""
        try:
            result = subprocess.run(['crontab', '-l'], capture_output=True, text=True, check=True)
            cron_content = result.stdout
            
            # Look for lines containing our script path
            script_path = os.path.abspath(__file__)
            for line in cron_content.split('\n'):
                if script_path in line and not line.strip().startswith('#'):
                    return line.strip()
            return None
            
        except subprocess.CalledProcessError:
            # No crontab exists yet
            return None
        except Exception as e:
            self.logger.error(f"Error reading crontab: {e}")
            return None
    
    def update_cron_schedule(self) -> bool:
        """Update the cron schedule based on current configuration"""
        try:
            # Get current crontab
            try:
                result = subprocess.run(['crontab', '-l'], capture_output=True, text=True, check=True)
                current_cron = result.stdout
            except subprocess.CalledProcessError:
                current_cron = ""
            
            # Remove existing entries for this script
            script_path = os.path.abspath(__file__)
            config_path = os.path.abspath(self.config.config_file_path or "/etc/sai-inference/daily-test.env")
            
            lines = []
            removed_count = 0
            for line in current_cron.split('\n'):
                if line.strip() and script_path not in line:
                    lines.append(line)
                elif script_path in line:
                    removed_count += 1
                    self.logger.info(f"Removed existing cron entry: {line.strip()}")
            
            # Add new entry if enabled
            if self.config.cron_enabled:
                new_entry = f"{self.config.cron_schedule} {script_path} {config_path}"
                lines.append(new_entry)
                lines.append("")  # Add blank line at end
                self.logger.info(f"Added new cron entry: {new_entry}")
                self.logger.info(f"Schedule: {self.config.cron_description}")
            else:
                self.logger.info("Cron scheduling disabled - no entry added")
            
            # Update crontab
            new_cron = '\n'.join(lines)
            process = subprocess.run(['crontab', '-'], input=new_cron, text=True, check=True)
            
            if removed_count > 0 or self.config.cron_enabled:
                self.logger.info("✅ Cron schedule updated successfully")
            else:
                self.logger.info("✅ Cron schedule cleared")
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update cron schedule: {e}")
            return False
    
    def show_cron_status(self) -> Dict[str, Any]:
        """Show current cron configuration and status"""
        current_entry = self.get_current_cron_entry()
        
        status = {
            "config_enabled": self.config.cron_enabled,
            "config_schedule": self.config.cron_schedule,
            "config_description": self.config.cron_description,
            "current_cron_entry": current_entry,
            "is_scheduled": current_entry is not None,
            "needs_update": False
        }
        
        # Check if current cron matches config
        if current_entry and self.config.cron_enabled:
            expected_entry = f"{self.config.cron_schedule} {os.path.abspath(__file__)}"
            if not current_entry.startswith(expected_entry):
                status["needs_update"] = True
        elif current_entry and not self.config.cron_enabled:
            status["needs_update"] = True
        elif not current_entry and self.config.cron_enabled:
            status["needs_update"] = True
            
        return status


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SAI Daily Test Service")
    parser.add_argument("config_file", nargs="?", help="Configuration file path")
    parser.add_argument("--update-cron", action="store_true", 
                       help="Update cron schedule based on configuration")
    parser.add_argument("--show-cron", action="store_true",
                       help="Show current cron configuration and status")
    parser.add_argument("--disable-cron", action="store_true",
                       help="Disable and remove cron schedule")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = DailyTestConfig(args.config_file)
        
        # Validate configuration
        errors = config.validate()
        if errors and not (args.show_cron):  # Allow show-cron even with validation errors
            print("Configuration errors:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        
        service = DailyTestService(config)
        
        # Handle cron management commands
        if args.update_cron:
            print("🔄 Updating cron schedule...")
            if service.update_cron_schedule():
                status = service.show_cron_status()
                if status["is_scheduled"]:
                    print(f"✅ Cron schedule set: {status['config_description']}")
                    print(f"   Entry: {status['current_cron_entry']}")
                else:
                    print("✅ Cron schedule disabled")
                return 0
            else:
                return 1
                
        elif args.show_cron:
            status = service.show_cron_status()
            print("📅 Cron Schedule Status:")
            print(f"   Config Enabled: {status['config_enabled']}")
            print(f"   Config Schedule: {status['config_schedule']}")
            print(f"   Description: {status['config_description']}")
            print(f"   Currently Scheduled: {status['is_scheduled']}")
            if status['current_cron_entry']:
                print(f"   Current Entry: {status['current_cron_entry']}")
            if status['needs_update']:
                print("   ⚠️ Configuration changed - run --update-cron to apply")
            return 0
            
        elif args.disable_cron:
            print("🔄 Disabling cron schedule...")
            # Temporarily disable cron in config for this operation
            config.cron_enabled = False
            service.config = config
            if service.update_cron_schedule():
                print("✅ Cron schedule disabled and removed")
                return 0
            else:
                return 1
        
        # Default: Run tests
        else:
            results = service.run_tests()
            
            # Exit with appropriate code
            if results['failed_tests'] > 0:
                return 1
            else:
                return 0
            
    except KeyboardInterrupt:
        print("Operation interrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Operation failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit(main())