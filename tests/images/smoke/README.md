# Smoke Detection Test Images

Place images with **known smoke detections** in this directory.

## Purpose
These images are used to test the smoke detection alert chain from:
1. Image → SAI Inference Service 
2. Detection → n8n Workflow
3. Alert → Notification Systems

## Image Requirements
- **Content**: Must contain visible smoke that SAI models can reliably detect
- **Format**: .jpg, .jpeg, .png, .bmp, .tiff, .webp
- **Resolution**: Recommended 1920px or higher for optimal SAI detection
- **Quality**: Clear, unambiguous smoke presence for consistent test results

## Test Behavior
- One image randomly selected per test run
- Sent to n8n webhook with metadata indicating test source
- Expected to trigger smoke detection alerts
- Results logged to `/var/log/sai-inference/daily-test.log`

## Adding Images
Simply copy smoke detection images to this directory:

```bash
cp your_smoke_image.jpg /path/to/sai-inference/tests/images/smoke/
```

Multiple images provide test variety.