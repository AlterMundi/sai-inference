# Fire and Smoke Detection Test Images

Place images with **both fire and smoke detections** in this directory.

## Purpose
These images are used to test combined detection alert chains from:
1. Image → SAI Inference Service 
2. Combined Detection → n8n Workflow
3. High-Priority Alerts → Notification Systems

## Image Requirements
- **Content**: Must contain both visible fire AND smoke that SAI models can reliably detect
- **Format**: .jpg, .jpeg, .png, .bmp, .tiff, .webp
- **Resolution**: Recommended 1920px or higher for optimal SAI detection
- **Quality**: Clear, unambiguous presence of both fire and smoke

## Test Behavior
- One image randomly selected per test run
- Sent to n8n webhook with metadata indicating test source
- Expected to trigger both fire AND smoke detection alerts
- May trigger high-priority/emergency alert paths
- Results logged to `/var/log/sai-inference/daily-test.log`

## Adding Images
Simply copy combined detection images to this directory:

```bash
cp your_fire_and_smoke_image.jpg /path/to/sai-inference/tests/images/both/
```

Multiple images provide test variety and reduce cache effects.

## Special Considerations
Images in this directory typically trigger the highest priority alerts since they indicate both types of hazards simultaneously. Ensure your n8n workflows and notification systems are properly configured for these combined detection scenarios.