# Fire Detection Test Images

Place images with **known fire detections** in this directory.

## Purpose
These images are used to test the fire detection alert chain from:
1. Image → SAI Inference Service 
2. Detection → n8n Workflow
3. Alert → Notification Systems

## Image Requirements
- **Content**: Must contain visible fire that SAI models can reliably detect
- **Format**: .jpg, .jpeg, .png, .bmp, .tiff, .webp
- **Resolution**: Recommended 1920px or higher for optimal SAI detection
- **Quality**: Clear, unambiguous fire presence for consistent test results

## Test Behavior
- One image randomly selected per test run
- Sent to n8n webhook with metadata indicating test source
- Expected to trigger fire detection alerts
- Results logged to `/var/log/sai-inference/daily-test.log`

## Adding Images
Simply copy fire detection images to this directory:

```bash
cp your_fire_image.jpg /path/to/sai-inference/tests/images/fire/
```

Multiple images provide test variety.