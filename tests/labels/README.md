# SAI Test Labels Directory

YOLO format annotation files for test images. Must match image filenames exactly (e.g., `test.jpg` → `test.txt`).

## Structure
```
tests/labels/
├── fire/     # Fire detection annotations (class 0)
├── smoke/    # Smoke detection annotations (class 1)  
└── both/     # Combined detection annotations
```

## YOLO Format
```
class_id x_center y_center width height [confidence]
```

**Class IDs:** 0=fire, 1=smoke  
**Coordinates:** Normalized 0.0-1.0

## Examples

**Fire:** `0 0.5 0.4 0.3 0.2`  
**Smoke:** `1 0.45 0.35 0.4 0.3`  
**Combined:** 
```
0 0.3 0.5 0.25 0.2
1 0.6 0.4 0.35 0.25
```

## Generated Metadata

Annotations are parsed and added to n8n requests:
```json
{
  "yolo_annotations": [...],
  "annotation_count": 3,
  "detected_classes": ["fire", "smoke"],
  "has_fire": true,
  "has_smoke": true
}
```