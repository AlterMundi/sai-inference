#!/usr/bin/env python3
"""
Test configuration after reference implementation updates
"""

# Import without full app dependencies
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import Settings

def main():
    settings = Settings()
    
    print("🔥 SAINet2.1 Reference Configuration Test")
    print("=" * 50)
    
    # Test critical parameters
    tests = [
        ("Confidence Threshold", settings.model_confidence, 0.15, "Reference: conf=0.15"),
        ("Input Resolution", settings.input_size, 1920, "Reference: imgsz=1920"),
        ("IoU Threshold", settings.model_iou_threshold, 0.45, "Standard NMS"),
        ("Default Model", settings.default_model, "sai_v2.1.pt", "SAINet2.1"),
    ]
    
    all_passed = True
    
    for test_name, actual, expected, note in tests:
        status = "✅" if actual == expected else "❌"
        if actual != expected:
            all_passed = False
        print(f"{status} {test_name}: {actual} (expected: {expected}) - {note}")
    
    print("\n📊 Configuration Summary:")
    print(f"  • Model: {settings.default_model}")
    print(f"  • Resolution: {settings.input_size}px (vs 896px previous)")
    print(f"  • Confidence: {settings.model_confidence} (vs 0.45 previous)")
    print(f"  • Device: {settings.model_device}")
    print(f"  • API Port: {settings.port}")
    print(f"  • Webhook: {settings.n8n_webhook_path}")
    
    print("\n🎯 Impact of Changes:")
    print("  • 3x MORE SENSITIVE fire detection (0.15 vs 0.45)")
    print("  • 4.6x HIGHER RESOLUTION analysis (1920² vs 896²)")
    print("  • Better small fire/smoke detection")
    print("  • Optimized for life-safety applications")
    
    if all_passed:
        print("\n🎉 All configuration tests PASSED! Ready for SAINet2.1.")
        return 0
    else:
        print("\n⚠️  Some configuration tests FAILED! Check implementation.")
        return 1

if __name__ == "__main__":
    exit(main())