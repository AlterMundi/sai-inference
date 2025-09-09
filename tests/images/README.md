# SAI Test Images Directory

This directory contains test images for the daily test service, organized by detection type.

## Directory Structure

```
tests/images/
├── fire/     # Images with known fire detections
├── smoke/    # Images with known smoke detections
├── both/     # Images with both fire and smoke detections
└── README.md # This file
```

## Usage

The daily test service (`src/daily_test.py`) automatically discovers images in these subdirectories and sends them to the configured n8n webhook for end-to-end alert system validation.

### Adding Test Images

1. **Fire Detection Images**: Place in `fire/` subdirectory
   - Should contain visible fire that SAI models can detect
   - Will test fire detection alert chains

2. **Smoke Detection Images**: Place in `smoke/` subdirectory
   - Should contain visible smoke that SAI models can detect  
   - Will test smoke detection alert chains

3. **Combined Detection Images**: Place in `both/` subdirectory
   - Should contain both fire and smoke detections
   - Will test combined alert scenarios

### Supported Formats

- `.jpg`, `.jpeg` (recommended)
- `.png`
- `.bmp`
- `.tiff`, `.tiff`
- `.webp`

### Image Requirements

- **Resolution**: Images should be reasonably high resolution (recommended: 1920px or higher)
- **Content**: Must contain detectable fire/smoke for positive test results
- **Quality**: Clear, unambiguous detections work best for consistent testing

### Test Selection

- The service randomly selects one image from each enabled category per test run
- Multiple images per category provide test variety and reduce cache effects
- At least one image per enabled category is required

## Configuration

Test categories can be enabled/disabled in the configuration file:

```env
ENABLE_FIRE_TESTS=true
ENABLE_SMOKE_TESTS=true  
ENABLE_BOTH_TESTS=true
```

## Manual Testing

Test the service manually:

```bash
# Run single test
python src/daily_test.py

# Run with specific config
python src/daily_test.py config/daily-test.env
```